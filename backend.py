import random
import json
import os
import re
import subprocess
import time
import sys
import shutil
import logging
import requests
import numpy as np
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydub import AudioSegment
import whisper
from datetime import datetime, timedelta, timezone

from google import genai
from google.genai import types as genai_types

# ==========================================
# DIRECTORIES & CONSTANTS
# ==========================================
WORKING_DIR = "./working"
INPUT_DIR = "./input"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(INPUT_DIR, exist_ok=True)

USER_ID = 1 # Universal ID for the single web admin user

# ==========================================
# LOG SILENCER
# ==========================================
class SuppressKaggleLogs:
    def __enter__(self):
        self.devnull = open(os.devnull, 'w')
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        sys.stdout = self.devnull
        sys.stderr = self.devnull
        logging.disable(logging.CRITICAL)
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.devnull.close()
        logging.disable(logging.NOTSET)

# ==========================================
# ENVIRONMENT VARIABLES & SECRETS
# ==========================================
PROMPT = os.environ.get("PROMPT", "").strip()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GIST_ID = os.environ.get("GIST_ID", "").strip()
GIST_TOKEN = os.environ.get("GIST_TOKEN", "").strip()

if not GEMINI_API_KEY:
    print("❌ ERROR: GEMINI_API_KEY missing in GitHub Secrets!")
    sys.exit(1)
if not PROMPT:
    print("❌ ERROR: No prompt provided!")
    sys.exit(1)

API_KEYS = [GEMINI_API_KEY]
class SmartAPIKeyManager:
    def __init__(self, keys):
        self.available_keys = keys.copy()
        self.lock = threading.RLock()
    def get_random_key(self):
        with self.lock:
            if not self.available_keys: return None
            return random.choice(self.available_keys)
    def get_available_count(self):
        with self.lock: return len(self.available_keys)

api_manager = SmartAPIKeyManager(API_KEYS)

# State Management for Gist Database
batch_registry = {}       
chat_lang_counters = {}   
chat_history = {} 
user_settings = {
    USER_ID: {
        "bgm_volume": 7, "dataset": ["mix"], "bgm_dataset": ["mix"], 
        "voice": "Puck", "long_tts": False, "long_tts_size": 1500, 
        "overlay_on": True, "overlay_opacity": 75, "visual_cuts": 0, 
        "sub_pos": "center", "sub_bg": False
    }
}
state_lock = threading.RLock()

# ==========================================
# GITHUB GIST CLOUD DATABASE LOGIC
# ==========================================
def clean_old_history(h_list):
    now_ist = datetime.now(timezone(timedelta(hours=5, minutes=30))).replace(tzinfo=None)
    valid = []
    for h in h_list:
        try:
            dt = datetime.strptime(h.get('time', ''), "%d %b %y, %I:%M %p")
            if (now_ist - dt).days <= 7: valid.append(h)
        except: valid.append(h)
    return valid[:500]

def load_history():
    global batch_registry, chat_lang_counters, chat_history, user_settings
    if not GIST_ID or not GIST_TOKEN: return
    headers = {"Authorization": f"Bearer {GIST_TOKEN}", "Accept": "application/vnd.github+json"}
    try:
        resp = requests.get(f"https://api.github.com/gists/{GIST_ID}", headers=headers, timeout=15)
        if resp.status_code == 200:
            files = resp.json().get("files", {})
            if "database.json" in files:
                content = files["database.json"]["content"]
                if content.strip() == "{}" or not content: return 
                data = json.loads(content)
                with state_lock:
                    batch_registry.update(data.get("registry", {}))
                    for cid_str, counters in data.get("counters", {}).items(): chat_lang_counters[int(cid_str)] = counters
                    for cid_str, s in data.get("settings", {}).items(): user_settings[int(cid_str)] = s
                    for cid_str, h_list in data.get("history", {}).items(): chat_history[int(cid_str)] = clean_old_history(h_list)
                print("✅ Database loaded successfully from Gist!")
    except Exception as e: print(f"Database Load Error: {e}")

def save_history():
    if not GIST_ID or not GIST_TOKEN: return
    try:
        with state_lock:
            data = {"registry": batch_registry, "counters": chat_lang_counters, "settings": user_settings, "history": chat_history}
            content = json.dumps(data)
        headers = {"Authorization": f"Bearer {GIST_TOKEN}", "Accept": "application/vnd.github+json"}
        payload = {"files": {"database.json": {"content": content}}}
        requests.patch(f"https://api.github.com/gists/{GIST_ID}", headers=headers, json=payload, timeout=15)
        print("✅ Database successfully saved to Gist!")
    except Exception as e: print(f"Database Save Error: {e}")

# ==========================================
# AUDIO GENERATION
# ==========================================
def chunk_text(text, max_chars):
    words = text.split()
    chunks, curr, curr_len = [], [], 0
    for w in words:
        if curr_len + len(w) + 1 > max_chars and curr:
            chunks.append(" ".join(curr))
            curr, curr_len = [w], len(w)
        else:
            curr.append(w)
            curr_len += len(w) + 1
    if curr: chunks.append(" ".join(curr))
    return chunks if chunks else [text]

def generate_audio_with_gemini(text, voice="Puck"):
    api_key = api_manager.get_random_key()
    if not api_key: raise Exception("No API key available")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-tts", contents=text,
        config=genai_types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=genai_types.SpeechConfig(
                voice_config=genai_types.VoiceConfig(prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(voice_name=voice)))))
    return response.candidates[0].content.parts[0].inline_data.data

def save_audio_to_file(audio_bytes, filename):
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
    audio_float = audio_array.astype(np.float32) / 32768.0
    sf.write(filename, audio_float, 24000)

# ==========================================
# DATASET UTILS
# ==========================================
def scan_multi_datasets(input_dir=INPUT_DIR):
    dataset_map = {}
    if not os.path.exists(input_dir): return dataset_map
    for root, _, files in os.walk(input_dir):
        videos = [os.path.join(root, f) for f in files if f.lower().endswith(('.mp4', '.mov', '.mkv', '.webm'))]
        if videos:
            folder_name = os.path.basename(root)[:35].strip() or "Root_Videos"
            if folder_name in dataset_map: dataset_map[folder_name].extend(videos)
            else: dataset_map[folder_name] = videos
    return dataset_map

def scan_bgm_datasets(input_dir=INPUT_DIR):
    dataset_map = {}
    if not os.path.exists(input_dir): return dataset_map
    for root, _, files in os.walk(input_dir):
        bgms = [os.path.join(root, f) for f in files if f.lower().endswith(('.mp3', '.wav', '.m4a')) and "voiceover" not in f.lower()]
        if bgms:
            folder_name = os.path.basename(root)[:35].strip() or "Root_Audio"
            if folder_name in dataset_map: dataset_map[folder_name].extend(bgms)
            else: dataset_map[folder_name] = bgms
    return dataset_map

def get_video_specs(file_path):
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,duration', '-of', 'json', file_path]
    output = json.loads(subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode('utf-8').strip())
    stream = output.get('streams', [{}])[0]
    return float(stream.get('duration', output.get('format', {}).get('duration', 0.0))), max(2, int(stream.get('width', 1920))), max(2, int(stream.get('height', 1080)))

def get_media_duration(file_path):
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', file_path]
    return float(subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode('utf-8').strip())

# ==========================================
# METADATA & PROCESSING
# ==========================================
def generate_title_and_hashtags(transcribed_text):
    if not transcribed_text or len(transcribed_text.strip()) < 3: return "Cinematic Video\n#Country #Language"
    words = transcribed_text.split()
    first_line_title = " ".join(words[:10]) + ("..." if len(words) > 10 else "")
    return f"{first_line_title}\n#Country #Language"

def standardize_clip_task(idx, clip, job_id, overlay_on, overlay_op):
    norm_output = f"/tmp/norm_{job_id}_{idx}.mp4"
    iw, ih = clip["w"], clip["h"]
    sw, sh = (int(iw * (1920.0 / ih)) if (iw * (1920.0 / ih)) >= 1080 else 1080), (1920 if (iw * (1920.0 / ih)) >= 1080 else int(ih * (1080.0 / iw)))
    sw, sh = (int(sw * clip["zoom"]) // 2) * 2, (int(sh * clip["zoom"]) // 2) * 2
    crop_x_max, crop_y_max = max(0, sw - 1080), max(0, sh - 1920)
    crop_x = random.randint(5, max(5, crop_x_max - 5))
    crop_y = random.randint(5, max(5, crop_y_max - 5))
    
    filter_str = f"trim=start={clip['start_time']}:duration={clip['desired_orig']},setpts=PTS-STARTPTS,setpts={1.0 / clip['speed']:.4f}*PTS,scale={sw}:{sh},crop=1080:1920:'max(0,min({crop_x}+3*sin(2*PI*t*0.6),{crop_x_max}))':'max(0,min({crop_y}+3*cos(2*PI*t*0.6),{crop_y_max}))',setsar=1"
    if clip["flip"]: filter_str += ",hflip"
    if overlay_on: filter_str += f",drawbox=color=black@{overlay_op/100.0}:t=fill,vignette=angle=0.3"
    filter_str += ",fps=30"
    
    cmd = ['ffmpeg', '-y', '-i', clip["file"], '-vf', filter_str, '-c:v', 'libx264', '-crf', '18', '-preset', 'fast', '-pix_fmt', 'yuv420p', '-r', '30', '-an', norm_output]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return norm_output if os.path.exists(norm_output) else None
    except: return None

def generate_ass_from_words(words_list, ass_path, sub_pos, sub_bg):
    align = "5" if sub_pos == "center" else "2"
    margin_v = "120" if sub_pos == "center" else "650"
    ass_header = f"[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\nScaledBorderAndShadow: yes\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Default,DejaVu Sans Light,68,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,0,0,{align},150,150,{margin_v},1\nStyle: SubBg,DejaVu Sans Light,68,&HFF000000,&HFF000000,&HFF000000,&H4C000000,-1,0,0,0,100,100,0,0,3,10,0,{align},150,150,{margin_v},1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    
    events_section, text_parts = "", []
    sentences, is_first = [], True
    word_index, total_words = 0, len(words_list)
    
    def format_ts(sec):
        sec = max(0.0, sec)
        return f"{int(sec//3600):01d}:{int((sec%3600)//60):02d}:{int(sec%60):02d}.{int(round((sec%1)*100)):02d}"

    while word_index < total_words:
        group_words = []
        g_start = words_list[word_index]['start']
        g_end = words_list[word_index]['end']
        while word_index < total_words:
            w = words_list[word_index]
            cw = w['word'].strip()
            if cw:
                group_words.append(cw)
                text_parts.append(cw)
            g_end = w['end']
            word_index += 1
            if cw:
                if is_first and (re.match(r'.*[.!?।]["\']?$', cw) or len(group_words) >= 11): break
                elif not is_first and (re.match(r'.*[.,\/#!$%\^&\*;:{}=\-_`~()।?!]["\']?$', cw) or len(group_words) >= 7): break
        if group_words:
            sentences.append({'start': g_start, 'end': g_end, 'text': ' '.join(group_words)})
            is_first = False

    if sentences:
        sentences[0]['start'] = 0.0
        if len(sentences) > 1:
            sentences[0]['end'] = max(sentences[0]['end'], 4.0)
            sentences = [sentences[0]] + [s for s in sentences[1:] if s['end'] > 4.0]
            for s in sentences[1:]:
                if s['start'] < 4.0: s['start'] = 4.0

    for i in range(len(sentences)):
        if i < len(sentences) - 1: sentences[i]['end'] = sentences[i+1]['start']
        if sentences[i]['end'] > sentences[i]['start']:
            t_str, s_ts, e_ts = sentences[i]['text'], format_ts(sentences[i]['start']), format_ts(sentences[i]['end'])
            if sub_bg:
                events_section += f"Dialogue: 0,{s_ts},{e_ts},SubBg,,0,0,0,,{{\\blur2}} {t_str} \nDialogue: 1,{s_ts},{e_ts},Default,,0,0,0,,{t_str}\n"
            else:
                events_section += f"Dialogue: 0,{s_ts},{e_ts},Default,,0,0,0,,{{\\1a&HFF&\\3c&H000000&\\bord12\\blur12}}{t_str}\nDialogue: 1,{s_ts},{e_ts},Default,,0,0,0,,{t_str}\n"

    with open(ass_path, "w", encoding="utf-8") as f: f.write(ass_header + events_section)
    return " ".join(text_parts)

# ==========================================
# MAIN EXECUTION ROUTINE
# ==========================================
def run_backend():
    print("🚀 Initializing Serverless Video Backend...")
    load_history()
    s = user_settings[USER_ID]
    
    # 1. KAGGLE FETCH COMMAND HANDLING
    if PROMPT.startswith("/fetch_dataset"):
        dataset_slug = PROMPT.replace("/fetch_dataset", "").strip().replace("https://www.kaggle.com/datasets/", "")
        print(f"📥 Downloading Dataset: {dataset_slug}...")
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            folder_name = dataset_slug.split("/")[-1]
            download_path = os.path.join(INPUT_DIR, folder_name)
            api.dataset_download_files(dataset_slug, path=download_path, unzip=True)
            print("✅ Dataset downloaded successfully!")
        except Exception as e: print(f"❌ Kaggle Download Error: {e}")
        return # End execution

    # 2. STANDARD SCRIPT PROCESSING
    print(f"🎙️ Generating AI Voiceover ({s['voice']})...")
    chunks = chunk_text(PROMPT, s["long_tts_size"]) if s["long_tts"] else [PROMPT]
    audio_segments = []
    
    for idx, chunk in enumerate(chunks):
        audio_data = generate_audio_with_gemini(chunk, s["voice"])
        temp_wav = f"/tmp/tts_{idx}.wav"
        save_audio_to_file(audio_data, temp_wav)
        audio_segments.append(AudioSegment.from_wav(temp_wav))
        
    final_audio = sum(audio_segments, AudioSegment.empty())
    temp_master_audio = "/tmp/master_audio.wav"
    final_audio.export(temp_master_audio, format="wav")

    print("🎧 Transcribing with Whisper...")
    whisper_model = whisper.load_model("base")
    result = whisper_model.transcribe(temp_master_audio, word_timestamps=True, initial_prompt="bingo bingo", condition_on_previous_text=False, fp16=False)
    
    cut_regions, all_words = [], []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            all_words.append(w)
            if is_bingo_trigger(w.get("word", "")): cut_regions.append((w["start"], w["end"]))
            
    merged_regions = []
    for st, en in cut_regions:
        if not merged_regions: merged_regions.append([st, en])
        else:
            if st - merged_regions[-1][1] <= 3.0: merged_regions[-1][1] = max(merged_regions[-1][1], en)
            else: merged_regions.append([st, en])
            
    current_pos_ms, valid_regions = 0, []
    for st, en in merged_regions:
        cut_st = max(0, int(st * 1000) - 400)
        if cut_st > current_pos_ms: valid_regions.append((current_pos_ms/1000.0, cut_st/1000.0))
        current_pos_ms = min(len(final_audio), int(en * 1000) + 400)
    if current_pos_ms < len(final_audio): valid_regions.append((current_pos_ms/1000.0, len(final_audio)/1000.0))
    if not valid_regions: valid_regions = [(0.0, len(final_audio)/1000.0)]

    print(f"✂️ Sliced into {len(valid_regions)} video(s).")
    
    # Process Video Render
    filtered_map = scan_multi_datasets() # Simplified: grab all downloaded videos
    bgm_map = scan_bgm_datasets()
    
    for i, (vs, ve) in enumerate(valid_regions):
        job_id = f"job_{int(time.time())}_{i}"
        chunk_seg = final_audio[int(vs * 1000):int(ve * 1000)]
        out_wav, out_ass = f"/tmp/{job_id}.wav", f"/tmp/{job_id}.ass"
        chunk_seg.export(out_wav, format="wav")
        
        c_words = [w for w in all_words if w.get("start", 0) >= vs and w.get("start", 0) < ve]
        for w in c_words: w["start"], w["end"] = max(0.0, w["start"] - vs), max(0.1, w["end"] - vs)
        
        full_text = generate_ass_from_words(c_words, out_ass, s["sub_pos"], s["sub_bg"])
        metadata = generate_title_and_hashtags(full_text)
        
        # Save meta.txt for GitHub Action Release notes
        with open(f"{WORKING_DIR}/meta_{job_id}.txt", "w") as f: f.write(metadata)
            
        audio_dur = ve - vs
        clips, _ = build_random_background_multi_dataset(filtered_map, audio_dur, s["visual_cuts"])
        
        music_pool = [m for k, v in bgm_map.items() for m in v]
        bgm_path = build_background_music(music_pool, audio_dur, job_id) if music_pool else None

        print(f"⚙️ Rendering Video {i+1}...")
        norm_files = []
        with ThreadPoolExecutor(max_workers=2) as ext:
            futs = {ext.submit(standardize_clip_task, idx, clip, job_id, s["overlay_on"], s["overlay_opacity"]): idx for idx, clip in enumerate(clips)}
            for f in as_completed(futs):
                if res := f.result(): norm_files.append(res)
                
        if not norm_files: continue
            
        final_mp4 = f"{WORKING_DIR}/final_{job_id}.mp4"
        cmd = ['ffmpeg', '-y']
        for f in norm_files: cmd.extend(['-i', f])
        cmd.extend(['-i', out_wav])
        
        v_filters, last_node, offset = [], "0:v", 0.0
        actual_durs = [get_media_duration(f) for f in norm_files]
        
        for idx in range(1, len(norm_files)):
            prev = max(0.6, actual_durs[idx-1])
            offset += (prev - 0.5)
            n_node = f"v{idx}"
            v_filters.append(f"[{last_node}][{idx}:v]xfade=transition=fade:duration=0.5:offset={offset:.3f}[{n_node}]")
            last_node = n_node
            
        v_str = ";".join(v_filters) + (";" if v_filters else "") + f"[{last_node}]subtitles={out_ass.replace(':', '\\:')}[fv]"
        
        tts_idx = len(norm_files)
        if bgm_path:
            cmd.extend(['-i', bgm_path])
            a_str = f"[{tts_idx}:a]volume=1.0[va];[{tts_idx+1}:a]volume={s['bgm_volume']/100.0:.2f}[ba];[va][ba]amix=inputs=2:duration=first:dropout_transition=0[fa]"
        else:
            a_str = f"[{tts_idx}:a]volume=1.0[fa]"
            
        cmd.extend(['-filter_complex', f"{v_str};{a_str}", '-map', '[fv]', '-map', '[fa]', '-c:v', 'libx264', '-preset', 'fast', '-crf', '26', '-maxrate', '1.2M', '-bufsize', '2.4M', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '64k', '-t', str(audio_dur), final_mp4])
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Cleanup temps
        for f in norm_files + [out_wav, out_ass, bgm_path]: 
            if f and os.path.exists(f): os.remove(f)

    # 3. Update Database History
    chat_h = chat_history.setdefault(USER_ID, [])
    chat_h.insert(0, {"batch_id": str(int(time.time())), "name": "Web Job", "count": len(valid_regions), "time": datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%d %b %y, %I:%M %p")})
    chat_history[USER_ID] = clean_old_history(chat_h)
    save_history()
    print("✅ Run Complete! File saved to Working Directory.")

if __name__ == "__main__":
    run_backend()

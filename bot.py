import telebot
from telebot import types
import random
import json
import io
import os
import re
import queue
import threading
import subprocess
import time
import sys
import shutil
import wave
import glob
import logging
from contextlib import redirect_stdout, redirect_stderr
import numpy as np
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydub import AudioSegment
import whisper
from whisper.tokenizer import LANGUAGES
from datetime import datetime, timedelta, timezone

from google import genai
from google.genai import types as genai_types

# ==========================================
# GITHUB ACTIONS DIRECTORIES FIX
# ==========================================
WORKING_DIR = "./working"
INPUT_DIR = "./input"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(INPUT_DIR, exist_ok=True)

# ==========================================
# ABSOLUTE LOG SILENCER
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
# GEMINI API SECRETS MANAGER
# ==========================================
API_KEYS = [] # User provides keys dynamically via Bot Settings UI
class SmartAPIKeyManager:
    def __init__(self, keys):
        self.available_keys = keys.copy()
        self.failed_keys = []
        self.lock = threading.RLock()
    def get_random_key(self):
        with self.lock:
            if not self.available_keys: return None
            return random.choice(self.available_keys)
    def mark_failed(self, key):
        with self.lock:
            if key in self.available_keys:
                self.available_keys.remove(key)
                self.failed_keys.append(key)
    def get_available_count(self):
        with self.lock: return len(self.available_keys)

api_manager = SmartAPIKeyManager(API_KEYS)
user_api_managers = {}  

# ==========================================
# LLM BENCHMARK SETUP
# ==========================================
os.environ['LLM_DEFAULT'] = 'anthropic/claude-sonnet-4'
import kaggle_benchmarks as kbench

# ==========================================
# KAGGLE DATASET HISTORY SETUP
# ==========================================
HISTORY_DIR = f"{WORKING_DIR}/history_data"
DATASET_SLUG = "vathsamajibail/data-beach"

TELEGRAM_BOT_TOKEN = "8538315366:AAEbO6mMCBxCOa1vTvN0JkZo2I5U0cS344I"
CHANNEL_ID = "-1003926894558"

print("🧠 Loading local offline Whisper-Base model on Runner CPU...")
whisper_model = whisper.load_model("base")
print("✅ Local Whisper Model Loaded and Ready!")

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

# GLOBAL QUEUE & JOB STATE
master_task_queue = queue.Queue() 
global_active_job = None  

jobs_by_chat = {}         
active_queue_messages = {} 
cancel_mode = {}          
ui_locks = {}
last_edit_times = {}
last_message_text = {} 

batch_registry = {}       
chat_lang_counters = {}   
chat_history = {} 
user_settings = {} 

state_lock = threading.RLock()
text_buffers = {}
buffer_lock = threading.RLock()

def get_ui_lock(chat_id):
    with state_lock:
        if chat_id not in ui_locks: ui_locks[chat_id] = threading.RLock()
        return ui_locks[chat_id]

def get_main_keyboard():
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    markup.add(types.KeyboardButton("📜 History"), types.KeyboardButton("⚙️ Settings"))
    return markup

def add_log(task, chat_id, msg):
    with state_lock:
        if "live_logs" not in task: task["live_logs"] = []
        task["live_logs"].append(msg)
        if len(task["live_logs"]) > 6: task["live_logs"].pop(0)
    update_queue_ui(chat_id)

def shorten_name(name):
    parts = name.split()
    if len(parts) >= 2: return f"{parts[0][:3]} {parts[1]}"
    return name[:6]

def clean_old_history(h_list):
    now_ist = datetime.now(timezone(timedelta(hours=5, minutes=30))).replace(tzinfo=None)
    valid = []
    for h in h_list:
        try:
            dt = datetime.strptime(h.get('time', ''), "%d %b %y, %I:%M %p")
            if (now_ist - dt).days <= 7: valid.append(h)
        except: valid.append(h)
    return valid[:500]

# ==========================================
# DYNAMIC KAGGLE DOWNLOADER (TELEGRAM COMMANDS)
# ==========================================
@bot.message_handler(commands=['set_kaggle'])
def handle_set_kaggle(message):
    parts = message.text.split()
    if len(parts) == 3:
        os.environ['KAGGLE_USERNAME'] = parts[1].strip()
        os.environ['KAGGLE_KEY'] = parts[2].strip()
        bot.reply_to(message, "✅ Kaggle credentials set successfully! You can now use `/fetch_dataset`.", parse_mode="Markdown")
        load_history()
    else:
        bot.reply_to(message, "⚠️ Usage: `/set_kaggle <username> <api_key>`", parse_mode="Markdown")

@bot.message_handler(commands=['fetch_dataset'])
def handle_fetch_dataset(message):
    parts = message.text.split()
    if len(parts) == 2:
        dataset_slug = parts[1].strip()
        dataset_slug = dataset_slug.replace("https://www.kaggle.com/datasets/", "")
        msg = bot.reply_to(message, f"⏳ Downloading dataset `{dataset_slug}` into GitHub Actions...\nThis may take a few minutes.", parse_mode="Markdown")
        
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            
            # Save into uniquely named folder
            folder_name = dataset_slug.split("/")[-1]
            download_path = os.path.join(INPUT_DIR, folder_name)
            os.makedirs(download_path, exist_ok=True)
            
            api.dataset_download_files(dataset_slug, path=download_path, unzip=True)
            bot.edit_message_text(f"✅ Dataset `{dataset_slug}` downloaded successfully!\nIt is now available in your video settings.", chat_id=msg.chat.id, message_id=msg.message_id, parse_mode="Markdown")
        except Exception as e:
            bot.edit_message_text(f"❌ Failed to download dataset. Did you `/set_kaggle` first?\nError: {str(e)}", chat_id=msg.chat.id, message_id=msg.message_id)
    else:
        bot.reply_to(message, "⚠️ Usage: `/fetch_dataset <username/dataset-slug>`", parse_mode="Markdown")

# ==========================================
# GEMINI AUDIO & API LOGIC
# ==========================================
def generate_audio_with_gemini(text, voice="Puck", model="gemini-2.5-flash-preview-tts", chat_id=None, max_attempts=15, on_api_fail=None):
    user_mgr = user_api_managers.get(chat_id) if chat_id else None
    attempt = 0
    while attempt < max_attempts:
        api_key = None
        is_user_key = False
        if user_mgr and user_mgr.get_available_count() > 0:
            api_key = user_mgr.get_random_key()
            is_user_key = True
        elif api_manager.get_available_count() > 0:
            api_key = api_manager.get_random_key()
            is_user_key = False
        if not api_key: break
            
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=model, contents=text,
                config=genai_types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=genai_types.SpeechConfig(
                        voice_config=genai_types.VoiceConfig(prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(voice_name=voice)))))
            return response.candidates[0].content.parts[0].inline_data.data, "SUCCESS"
        except Exception as e:
            if any(x in str(e).lower() for x in ['quota', 'limit', '429', '403']): 
                if is_user_key: user_mgr.mark_failed(api_key)
                else: api_manager.mark_failed(api_key)
            if on_api_fail: on_api_fail(api_key, str(e))
            time.sleep(0.5)
        attempt += 1
    if user_mgr and user_mgr.get_available_count() == 0 and len(user_settings.get(chat_id, {}).get("api_keys", [])) > 0:
        return None, "API_FAIL_USER"
    return None, "API_FAIL_DEFAULT"

def save_audio_to_file(audio_bytes, filename):
    try:
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = audio_array.astype(np.float32) / 32768.0
        sf.write(filename, audio_float, 24000)
        return True, len(audio_array) / 24000
    except:
        return False, 0

def load_history():
    global batch_registry, chat_lang_counters, chat_history, user_settings, user_api_managers
    if not os.environ.get('KAGGLE_USERNAME') or not os.environ.get('KAGGLE_KEY'):
        return # Skip loading if no credentials
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        with SuppressKaggleLogs():
            api = KaggleApi()
            api.authenticate()
            os.makedirs(HISTORY_DIR, exist_ok=True)
            api.dataset_download_files(DATASET_SLUG, path=HISTORY_DIR, unzip=True)
        for file in os.listdir(HISTORY_DIR):
            if file == "history.json" or file == "registry.json":
                with open(os.path.join(HISTORY_DIR, file), "r") as f: data = json.load(f)
                with state_lock:
                    batch_registry.update(data.get("registry", {}))
                    for cid_str, counters in data.get("counters", {}).items(): chat_lang_counters[int(cid_str)] = counters
                    for cid_str, s in data.get("settings", {}).items(): user_settings[int(cid_str)] = s
                    if "history" in data:
                        for cid_str, h_list in data.get("history", {}).items(): chat_history[int(cid_str)] = clean_old_history(h_list)
            elif file.startswith("user_") and file.endswith(".json"):
                chat_id = int(file.split("_")[1].split(".")[0])
                with open(os.path.join(HISTORY_DIR, file), "r") as f:
                    with state_lock: chat_history[chat_id] = clean_old_history(json.load(f))
        for cid, s in user_settings.items():
            if "api_keys" in s and s["api_keys"]:
                user_api_managers[cid] = SmartAPIKeyManager(s["api_keys"])
    except Exception as e: 
        print("Dataset History Error:", e)

def save_history_async():
    if not os.environ.get('KAGGLE_USERNAME') or not os.environ.get('KAGGLE_KEY'): return
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        with SuppressKaggleLogs():
            api = KaggleApi()
            api.authenticate()
            if os.path.exists(HISTORY_DIR): shutil.rmtree(HISTORY_DIR)
            os.makedirs(HISTORY_DIR, exist_ok=True)
            
            with state_lock:
                reg_data = {"registry": batch_registry, "counters": chat_lang_counters, "settings": user_settings}
                with open(os.path.join(HISTORY_DIR, "registry.json"), "w") as f: json.dump(reg_data, f)
                for chat_id, hist in chat_history.items():
                    with open(os.path.join(HISTORY_DIR, f"user_{chat_id}.json"), "w") as f: json.dump(hist, f)
            meta = {"title": "data-beach", "id": DATASET_SLUG, "licenses": [{"name": "CC0-1.0"}]}
            with open(os.path.join(HISTORY_DIR, "dataset-metadata.json"), "w") as f: json.dump(meta, f)
            
            api.dataset_create_version(HISTORY_DIR, version_notes="Auto-update", dir_mode="zip", quiet=True)
    except: pass

def trigger_save(): threading.Thread(target=save_history_async, daemon=True).start()

# ==========================================
# PROCESSING UTILITIES
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

def is_bingo_trigger(word_text):
    return re.sub(r'[^a-zA-Z0-9]', '', word_text).lower() in ["bingo", "bingos", "bingobingo"]

# ==========================================
# UPDATED HASHTAG & TITLE ENGINE
# ==========================================
def generate_title_and_hashtags(transcribed_text):
    words = transcribed_text.split()
    first_line_title = " ".join(words[:10]) + ("..." if len(words) > 10 else "")
    fallback_output = f"{first_line_title}\n#Country #Language"
    
    if not transcribed_text or len(transcribed_text.strip()) < 3: 
        return fallback_output
        
    # Check if Kaggle credentials are set (required for kbench LLM to deduce country)
    if not os.environ.get('KAGGLE_USERNAME') or not os.environ.get('KAGGLE_KEY'):
        return fallback_output
        
    prompt = (
        f"Text: '{transcribed_text}'\n"
        f"Task: On line 1, output the exact first sentence (or first 10 words) of the text to use as the video title. "
        f"On line 2, output exactly 2 hashtags ONLY: the specific Country and the specific Language related to the text. "
        f"Do not include the first 3 topic hashtags anymore. NO emojis. ONLY two lines total."
    )
    for _ in range(3):
        try:
            response = kbench.llm.prompt(prompt).replace('```', '').replace('<', '').replace('>', '').strip()
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            if len(lines) >= 2: 
                return f"{lines[0]}\n{lines[-1]}"
        except: 
            time.sleep(2)
            
    return fallback_output

def scan_multi_datasets(input_dir=INPUT_DIR):
    dataset_map = {}
    if not os.path.exists(input_dir): return dataset_map
    for root, _, files in os.walk(input_dir):
        videos = [os.path.join(root, f) for f in files if f.lower().endswith(('.mp4', '.mov', '.mkv', '.webm'))]
        if videos:
            folder_name = os.path.basename(root)[:35].strip()
            if not folder_name: folder_name = "Root_Videos"
            if folder_name in dataset_map: dataset_map[folder_name].extend(videos)
            else: dataset_map[folder_name] = videos
    return dataset_map

def scan_bgm_datasets(input_dir=INPUT_DIR):
    dataset_map = {}
    if not os.path.exists(input_dir): return dataset_map
    for root, _, files in os.walk(input_dir):
        bgms = [os.path.join(root, f) for f in files if f.lower().endswith(('.mp3', '.wav', '.m4a')) and "voiceover" not in f.lower()]
        if bgms:
            folder_name = os.path.basename(root)[:35].strip()
            if not folder_name: folder_name = "Root_Audio"
            if folder_name in dataset_map: dataset_map[folder_name].extend(bgms)
            else: dataset_map[folder_name] = bgms
    return dataset_map

def get_video_specs(file_path):
    try:
        cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,duration', '-of', 'json', file_path]
        output = json.loads(subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode('utf-8').strip())
        stream = output.get('streams', [{}])[0]
        w, h = max(2, int(stream.get('width', 1920))), max(2, int(stream.get('height', 1080)))
        dur = float(stream.get('duration', output.get('format', {}).get('duration', 0.0)))
        return dur, w, h
    except: return 0.0, 1920, 1080

def get_media_duration(file_path):
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', file_path]
        return float(subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode('utf-8').strip())
    except: return 0.0

def build_random_background_multi_dataset(dataset_map, target_duration, vis_cuts=0):
    selected_clips, accumulated_duration = [], 0.0
    if not dataset_map: return selected_clips, 0.0
        
    active_datasets = list(dataset_map.keys())
    for d in active_datasets: random.shuffle(dataset_map[d])
    dataset_index_map = {d: 0 for d in active_datasets}
    safe_target = target_duration + 1.5 
    XFADE_DUR = 0.5 
    
    while accumulated_duration < safe_target:
        random.shuffle(active_datasets)
        added_in_round = False
        for d in active_datasets:
            idx = dataset_index_map[d]
            if idx < len(dataset_map[d]):
                video_file = dataset_map[d][idx]
                dataset_index_map[d] += 1
                dur, w, h = get_video_specs(video_file)
                
                if dur > 1.5:
                    if vis_cuts == 0: target_clip_dur = random.uniform(4.0, 8.0)
                    else: target_clip_dur = float(vis_cuts)
                        
                    speed = random.uniform(0.85, 1.15)
                    desired_orig = target_clip_dur * speed
                    
                    if dur > desired_orig: start_time = random.uniform(0, dur - desired_orig)
                    else:
                        start_time = 0.0
                        desired_orig = dur
                        target_clip_dur = dur / speed
                        
                    zoom = random.uniform(1.05, 1.25)
                    
                    selected_clips.append({
                        "file": video_file, "orig_dur": dur, "adjusted_dur": target_clip_dur, 
                        "speed": speed, "flip": random.choice([True, False]), 
                        "zoom": zoom, "w": w, "h": h, "start_time": start_time,
                        "desired_orig": desired_orig
                    })
                    
                    if len(selected_clips) == 1: accumulated_duration += target_clip_dur
                    else: accumulated_duration += (target_clip_dur - XFADE_DUR)
                    added_in_round = True
                    if accumulated_duration >= safe_target: break
        if not added_in_round: break
        
    return selected_clips, accumulated_duration

def build_background_music(music_files, target_duration, job_id):
    valid_music = [f for f in music_files if get_media_duration(f) > 5.0]
    if not valid_music: return None
    track = random.choice(valid_music)
    merged_music_path = f"/tmp/merged_music_{job_id}.mp3"
    cmd = [
        'ffmpeg', '-y', '-stream_loop', '-1', '-i', track, 
        '-af', 'loudnorm=I=-16:TP=-1.5:LRA=11',
        '-c:a', 'libmp3lame', '-b:a', '128k', '-ar', '44100', '-t', f'{target_duration:.2f}', 
        merged_music_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return merged_music_path
    except: return None

# ==========================================
# PARALLEL STANDARDIZATION WORKER
# ==========================================
def standardize_clip_task(idx, clip, job_id, overlay_on, overlay_op):
    norm_output = f"/tmp/norm_{job_id}_{idx}.mp4"
    iw, ih = clip["w"], clip["h"]
    sw, sh = (int(iw * (1920.0 / ih)) if (iw * (1920.0 / ih)) >= 1080 else 1080), (1920 if (iw * (1920.0 / ih)) >= 1080 else int(ih * (1080.0 / iw)))
    sw, sh = (int(sw * clip["zoom"]) // 2) * 2, (int(sh * clip["zoom"]) // 2) * 2
    crop_x_max, crop_y_max = max(0, sw - 1080), max(0, sh - 1920)
    crop_x = random.randint(5, max(5, crop_x_max - 5))
    crop_y = random.randint(5, max(5, crop_y_max - 5))
    
    filter_str = f"trim=start={clip['start_time']}:duration={clip['desired_orig']},setpts=PTS-STARTPTS,"
    filter_str += f"setpts={1.0 / clip['speed']:.4f}*PTS,"
    filter_str += f"scale={sw}:{sh},crop=1080:1920:'max(0,min({crop_x}+3*sin(2*PI*t*0.6),{crop_x_max}))':'max(0,min({crop_y}+3*cos(2*PI*t*0.6),{crop_y_max}))',setsar=1"
    
    if clip["flip"]: filter_str += ",hflip"
    if overlay_on: filter_str += f",drawbox=color=black@{overlay_op/100.0}:t=fill,vignette=angle=0.3"
    
    filter_str += (
        ",drawtext=text='•  •':x='mod(t*220,w+100)-100':y='mod(t*90+h*0.1,h+50)-50+40*sin(t*1.5)':fontcolor=white@0.65:fontsize=12,"
        "drawtext=text='•   •':x='mod(t*310,w+100)-100':y='mod(t*140+h*0.3,h+50)-50+60*sin(t*2)':fontcolor=white@0.45:fontsize=10,"
        "drawtext=text='• •':x='mod(t*160,w+100)-100':y='mod(t*70+h*0.5,h+50)-50+30*cos(t*1)':fontcolor=white@0.3:fontsize=6,"
        "drawtext=text='•    •':x='mod(t*360,w+100)-100':y='mod(t*160+h*0.7,h+50)-50+80*sin(t*2.5)':fontcolor=white@0.55:fontsize=10"
    )
    filter_str += f",fps=30"
    
    cmd = [
        'ffmpeg', '-y', '-i', clip["file"], '-vf', filter_str,
        '-c:v', 'libx264', '-crf', '18', '-preset', 'fast', '-pix_fmt', 'yuv420p', '-r', '30',
        '-an', norm_output
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if os.path.exists(norm_output) and os.path.getsize(norm_output) > 10000: return norm_output
        return None
    except subprocess.CalledProcessError as e: return None

def generate_ass_from_words(words_list, ass_path, sub_pos="center", sub_bg=False):
    align = "5" if sub_pos == "center" else "2"
    margin_v = "120" if sub_pos == "center" else "650"
    
    ass_header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans Light,68,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,0,0,{align},150,150,{margin_v},1
Style: SubBg,DejaVu Sans Light,68,&HFF000000,&HFF000000,&HFF000000,&H4C000000,-1,0,0,0,100,100,0,0,3,10,0,{align},150,150,{margin_v},1
\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"""

    events_section = ""
    total_words, word_index, text_parts = len(words_list), 0, []
    
    def format_ts(seconds):
        seconds = max(0.0, seconds)
        h, m, s, cs = int(seconds // 3600), int((seconds % 3600) // 60), int(seconds % 60), int(round((seconds % 1) * 100))
        if cs >= 100: s, cs = s + 1, 0
        return f"{h:01d}:{m:02d}:{s:02d}.{cs:02d}"

    sentences, is_first_line = [], True
    while word_index < total_words:
        group_words = []
        group_start = words_list[word_index]['start']
        group_end = words_list[word_index]['end']
        
        while word_index < total_words:
            w = words_list[word_index]
            clean_word = w['word'].strip()
            if clean_word:
                group_words.append(clean_word)
                text_parts.append(clean_word)
            group_end = w['end']
            word_index += 1
            if clean_word:
                if is_first_line:
                    if re.match(r'.*[.!?।]["\']?$', clean_word) or len(group_words) >= 11: break
                else:
                    if re.match(r'.*[.,\/#!$%\^&\*;:{}=\-_`~()।?!]["\']?$', clean_word) or len(group_words) >= 7: break
                
        if group_words:
            sentences.append({'start': group_start, 'end': group_end, 'text': ' '.join(group_words)})
            is_first_line = False

    if sentences:
        sentences[0]['start'] = 0.0
        if len(sentences) > 1:
            sentences[0]['end'] = max(sentences[0]['end'], 4.0)
            filtered_sentences = [sentences[0]]
            for i in range(1, len(sentences)):
                if sentences[i]['end'] <= 4.0: continue 
                if sentences[i]['start'] < 4.0: sentences[i]['start'] = 4.0 
                filtered_sentences.append(sentences[i])
            sentences = filtered_sentences

    for i in range(len(sentences)):
        if i < len(sentences) - 1: sentences[i]['end'] = sentences[i+1]['start']
        if sentences[i]['end'] > sentences[i]['start']:
            t_str = sentences[i]['text']
            s_ts = format_ts(sentences[i]['start'])
            e_ts = format_ts(sentences[i]['end'])
            
            if sub_bg:
                bg_str = f" {t_str} " 
                events_section += f"Dialogue: 0,{s_ts},{e_ts},SubBg,,0,0,0,,{{\\blur2}}{bg_str}\n"
                events_section += f"Dialogue: 1,{s_ts},{e_ts},Default,,0,0,0,,{t_str}\n"
            else:
                events_section += f"Dialogue: 0,{s_ts},{e_ts},Default,,0,0,0,,{{\\1a&HFF&\\3c&H000000&\\bord12\\blur12}}{t_str}\n"
                events_section += f"Dialogue: 1,{s_ts},{e_ts},Default,,0,0,0,,{t_str}\n"

    with open(ass_path, "w", encoding="utf-8") as f: f.write(ass_header + events_section)
    return " ".join(text_parts)

def prep_chunk_parallel(sj, vs, ve, all_words, audio_seg, output_dir, chat_id, sub_pos, sub_bg):
    try:
        with state_lock: sj["status_label"] = "📝 Meta"
        update_queue_ui(chat_id)
        
        job_id = sj["job_id"]
        chunk_seg = audio_seg[int(vs * 1000):int(ve * 1000)]
        audio_output_path = os.path.join(output_dir, f"audio_{job_id}.wav")
        chunk_seg.export(audio_output_path, format="wav")
        
        chunk_words = []
        for w in all_words:
            w_start, w_end = w.get("start", 0.0), w.get("end", 0.0)
            if w_start >= vs and w_start < ve:
                chunk_words.append({"word": w.get("word", ""), "start": max(0.0, w_start - vs), "end": max(0.1, w_end - vs)})
                
        ass_subtitle_path = os.path.join(output_dir, f"sub_{job_id}.ass")
        full_text = generate_ass_from_words(chunk_words, ass_subtitle_path, sub_pos, sub_bg)
        llm_metadata = generate_title_and_hashtags(full_text)
        
        with state_lock:
            sj["audio_output_path"] = audio_output_path
            sj["audio_duration"] = ve - vs
            sj["ass_subtitle_path"] = ass_subtitle_path
            sj["llm_metadata"] = llm_metadata
            sj["status_label"] = "⏳ Queue"
        update_queue_ui(chat_id)
        return True
    except Exception as e:
        with state_lock: sj["status_label"] = "❌ Meta Err"
        update_queue_ui(chat_id)
        return False

# ==========================================
# ULTRA CLEAN QUEUE UI ENGINE
# ==========================================
def _format_queue_text_unlocked(chat_id):
    global global_active_job
    jobs = jobs_by_chat.get(chat_id, [])
    if not jobs: return "📋 **Queue Empty**", None
        
    active_job = next((j for j in jobs if j["status"] in ["generating_audio", "scanning", "rendering"]), None)
    queued_jobs = [j for j in jobs if j["status"] == "queued_scan"]
    completed_jobs = [j for j in jobs if j["status"] in ["completed", "failed_master", "cancelled", "failed_api_user", "failed_api_default"] and not (j["status"] == "cancelled" and j.get("total_videos", 0) == 0)]
    
    is_busy_other = (global_active_job and global_active_job.get("chat_id") != chat_id)
    text = ""
    
    if active_job:
        s_name = shorten_name(active_job.get('display_name', '...'))
        t_v = active_job.get("total_videos", 1)
        text += f"▶ **PROCESSING: {s_name} ({t_v} Vids)**\n━━━━━━━━━━━━━\n"
    elif queued_jobs:
        text += "⏳ **WAITING IN QUEUE**\n━━━━━━━━━━━━━\n"

    if is_busy_other and not active_job:
        ahead_count = sum(1 for qj in list(master_task_queue.queue) if qj.get("chat_id") != chat_id)
        if global_active_job: ahead_count += 1
        text += f"⚠️ **Busy: {ahead_count} ahead**\n━━━━━━━━━━━━━\n"
        
    if active_job:
        if active_job["status"] == "generating_audio": text += "🎙️ `Phase: Audio Gen`\n"
        elif active_job["status"] == "scanning": text += "🎙️ `Phase: Audio Scan`\n"
        else:
            p = active_job.get("progress_pct", 0)
            c_v, f_v, t_v = active_job.get("completed_videos", 0), active_job.get("failed_videos", 0), active_job.get("total_videos", 1)
            p_bar = "█" * (p // 10) + "░" * (10 - (p // 10))
            if f_v > 0: text += f"├ 🔄 {c_v}/{t_v} | ❌ {f_v}\n"
            else: text += f"├ 🔄 Gen: {c_v}/{t_v} Vids\n"
            text += f"├ ⏳ [{p_bar}] {p}%\n"
            
        text += "\n📡 **Live Activity:**\n"
        if active_job.get("api_fails", 0) > 0: text += f"├ ⚠️ API Fails: {active_job['api_fails']}\n"
            
        if active_job["status"] == "rendering":
            for sj in active_job.get("sub_jobs", []): text += f"├ Vid {sj['vid_idx']}: {sj.get('status_label', '⏳ Wait')}\n"
        else:
            if active_job.get("live_logs"):
                for log in active_job["live_logs"]: text += f"├ {log}\n"
        text += "━━━━━━━━━━━━━\n"
            
    if queued_jobs:
        for i, q in enumerate(queued_jobs, 1): text += f"**Q{i}:** 📜 `Wait Voice`\n" if q.get("type") == "text" else f"**Q{i}:** 🎙️ `Wait Scan`\n"
        text += "━━━━━━━━━━━━━\n"

    fetch_buttons = []
    
    if completed_jobs:
        text += "✅ **COMPLETED**\n"
        for c in completed_jobs[-4:]:
            t_v, f_v, c_v = c.get('total_videos', 1), c.get('failed_videos', 0), c.get('completed_videos', 0)
            s_name = shorten_name(c.get('display_name', '...'))
            if c["status"] == "failed_api_user": text += f"❌ {s_name} - User API Err\n"
            elif c["status"] == "failed_api_default": text += f"❌ {s_name} - Bot API Err\n"
            elif c["status"] == "failed_master": text += f"❌ {s_name} - Master Err\n"
            elif c["status"] == "cancelled":
                text += f"🛑 {s_name} - Cancel ({c_v}/{t_v})\n"
                if c_v > 0: fetch_buttons.append(types.InlineKeyboardButton(f"📥 {s_name} ({c_v})", callback_data=f"fetch_{c['batch_id']}"))
            else:
                if f_v == 0 and c_v == t_v: text += f"📦 {s_name} ({t_v}) - ✅ OK\n"
                else: text += f"📦 {s_name} ({t_v}) - ⚠️ {f_v} Fail\n"
                if c_v > 0: fetch_buttons.append(types.InlineKeyboardButton(f"📥 {s_name} ({c_v})", callback_data=f"fetch_{c['batch_id']}"))
        text += "━━━━━━━━━━━━━\n"
                
    text = text.strip()
    if text.endswith("━━━━━━━━━━━━━"): text = text[:-13].strip()
    
    markup = types.InlineKeyboardMarkup(row_width=1)
    if cancel_mode.get(chat_id, False) and (active_job or queued_jobs):
        markup.add(types.InlineKeyboardButton("🛑 Cancel Current", callback_data="do_cancel_current"), types.InlineKeyboardButton("💥 Cancel Entire Queue", callback_data="do_cancel_all"), types.InlineKeyboardButton("❌ Nevermind", callback_data="do_cancel_abort"))
    elif active_job or queued_jobs: markup.add(types.InlineKeyboardButton("🛑 Stop / Cancel", callback_data="req_cancel_menu"))

    if fetch_buttons:
        row = []
        for btn in fetch_buttons[-4:]:
            row.append(btn)
            if len(row) == 2: markup.row(*row); row = []
        if row: markup.row(*row)
            
    return text.strip(), markup

def move_queue_to_bottom(chat_id):
    lock = get_ui_lock(chat_id)
    with lock:
        with state_lock:
            old_msg_id = active_queue_messages.get(chat_id)
            text_payload, markup = _format_queue_text_unlocked(chat_id)
        if old_msg_id:
            try: bot.delete_message(chat_id, old_msg_id)
            except: pass
        try:
            new_msg = bot.send_message(chat_id, text_payload, parse_mode="Markdown", reply_markup=markup)
            with state_lock:
                active_queue_messages[chat_id] = new_msg.message_id
                last_message_text[chat_id] = text_payload
                last_edit_times[chat_id] = time.time()
        except: pass

def update_queue_ui(chat_id, force=False):
    lock = get_ui_lock(chat_id)
    with lock:
        with state_lock:
            msg_id = active_queue_messages.get(chat_id)
            text_payload, markup = _format_queue_text_unlocked(chat_id)
        if not msg_id or text_payload == last_message_text.get(chat_id): return
        now = time.time()
        if not force and chat_id in last_edit_times and now - last_edit_times[chat_id] < 1.5: return
        last_edit_times[chat_id] = now
        last_message_text[chat_id] = text_payload
        try: bot.edit_message_text(chat_id=chat_id, message_id=msg_id, text=text_payload, parse_mode="Markdown", reply_markup=markup)
        except Exception as e: 
            if "Too Many Requests" in str(e): last_edit_times[chat_id] = time.time() + 5.0

# ==========================================
# BUTTON CALLBACK HANDLERS & ADVANCED SETTINGS
# ==========================================
def send_settings_ui(chat_id, message_id=None, menu="main"):
    with state_lock:
        if chat_id not in user_settings: user_settings[chat_id] = {"bgm_volume": 7, "dataset": ["mix"], "bgm_dataset": ["mix"], "voice": "Puck", "long_tts": False, "long_tts_size": 1500, "overlay_on": True, "overlay_opacity": 75, "visual_cuts": 0, "sub_pos": "center", "sub_bg": False}
        s = user_settings[chat_id]
        vol, voice = s.get("bgm_volume", 7), s.get("voice", "Puck")
        long_tts, chunk_sz = s.get("long_tts", False), s.get("long_tts_size", 1500)
        overlay_on, overlay_op = s.get("overlay_on", True), s.get("overlay_opacity", 75)
        vis_cuts, sub_pos, sub_bg = s.get("visual_cuts", 0), s.get("sub_pos", "center"), s.get("sub_bg", False)
        
        ds_selected, bgm_selected = s.get("dataset", ["mix"]), s.get("bgm_dataset", ["mix"])
        if not isinstance(ds_selected, list): ds_selected = [ds_selected]
        if not isinstance(bgm_selected, list): bgm_selected = [bgm_selected]
        ds_display = "🔀 Mix" if "mix" in ds_selected or not ds_selected else (f"📁 {ds_selected[0][:10]}" if len(ds_selected)==1 else f"📁 {len(ds_selected)} Sel")
        bgm_display = "🔀 Mix" if "mix" in bgm_selected or not bgm_selected else (f"🎵 {bgm_selected[0][:10]}" if len(bgm_selected)==1 else f"🎵 {len(bgm_selected)} Sel")

    if menu == "main":
        markup = types.InlineKeyboardMarkup(row_width=2)
        markup.row(types.InlineKeyboardButton(f"🎬 Vid Data: {ds_display}", callback_data="set_menu_dataset"), types.InlineKeyboardButton(f"🎵 Audio Data: {bgm_display}", callback_data="set_menu_bgmdataset"))
        markup.row(types.InlineKeyboardButton(f"🗣 Voice: {voice}", callback_data="set_menu_voice"), types.InlineKeyboardButton(f"🔊 BGM Vol: {vol}%", callback_data="set_menu_volume"))
        
        tts_text, overlay_text = "🟢 ON" if long_tts else "🔴 OFF", f"🟢 ON ({overlay_op}%)" if overlay_on else "🔴 OFF"
        vis_text, pos_text, bg_text = f"{vis_cuts}s" if vis_cuts > 0 else "Default", "Bottom" if sub_pos == "bottom" else "Center", "🟢 ON" if sub_bg else "🔴 OFF"
        
        markup.row(types.InlineKeyboardButton(f"⬛ Overlay: {overlay_text}", callback_data="set_menu_overlay"), types.InlineKeyboardButton(f"✂️ Vis Cuts: {vis_text}", callback_data="set_menu_viscuts"))
        markup.row(types.InlineKeyboardButton(f"📍 Sub Pos: {pos_text}", callback_data="toggle_sub_pos"), types.InlineKeyboardButton(f"💬 Sub BG: {bg_text}", callback_data="toggle_sub_bg"))
        markup.row(types.InlineKeyboardButton(f"🔑 API Keys", callback_data="set_menu_apikey"), types.InlineKeyboardButton(f"🎙️ Long TTS: {tts_text}", callback_data="toggle_long_tts"))
                   
        if long_tts: markup.add(types.InlineKeyboardButton(f"📏 Chunk Size: {chunk_sz} Chars ({chunk_sz/1000.0}m)", callback_data="set_menu_long_tts"))
        markup.add(types.InlineKeyboardButton("✅ Done / Save", callback_data="collapse"))
        text = f"⚙️ **Settings**\nConfigure your background and voice preferences below."

    elif menu == "viscuts":
        markup = types.InlineKeyboardMarkup(row_width=3)
        opts = [0, 3, 4, 5, 6, 7, 8, 9, 10]
        btns = [types.InlineKeyboardButton(f"{'✅ ' if vis_cuts == o else ''}{'Default' if o==0 else f'{o}s'}", callback_data=f"set_vc_{o}") for o in opts]
        markup.add(*btns)
        markup.row(types.InlineKeyboardButton("🔙 Back", callback_data="set_menu_main"))
        text = f"✂️ **Visual Cuts (Clip Length)**\nSelect exact duration to randomly slice video clips. 'Default' applies standard variable duration."
        
    elif menu == "overlay":
        markup = types.InlineKeyboardMarkup(row_width=3)
        markup.row(types.InlineKeyboardButton("➖ 5%", callback_data="overop_-5"), types.InlineKeyboardButton(f"⬛ {overlay_op}%", callback_data="overop_none"), types.InlineKeyboardButton("➕ 5%", callback_data="overop_+5"))
        markup.row(types.InlineKeyboardButton(f"Toggle: {'🟢 ON' if overlay_on else '🔴 OFF'}", callback_data="toggle_overlay"))
        markup.row(types.InlineKeyboardButton("🔙 Back", callback_data="set_menu_main"))
        text = f"⬛ **Black Overlay Settings**\nTurn the dark vignette overlay ON/OFF and adjust its transparency (Currently: {overlay_op}%)."

    elif menu == "volume":
        markup = types.InlineKeyboardMarkup(row_width=3)
        markup.row(types.InlineKeyboardButton("➖ 10%", callback_data="vol_-10"), types.InlineKeyboardButton(f"🔊 {vol}%", callback_data="vol_none"), types.InlineKeyboardButton("➕ 10%", callback_data="vol_+10"))
        markup.row(types.InlineKeyboardButton("➖ 1%", callback_data="vol_-1"), types.InlineKeyboardButton("🔙 Back", callback_data="set_menu_main"), types.InlineKeyboardButton("➕ 1%", callback_data="vol_+1"))
        text = f"🔊 **Volume Settings**\nAdjust the background music volume (Currently: {vol}%)."

    elif menu == "apikey":
        markup = types.InlineKeyboardMarkup(row_width=1)
        keys = user_settings.get(chat_id, {}).get("api_keys", [])
        valid_keys = user_api_managers.get(chat_id).get_available_count() if chat_id in user_api_managers else 0
        text = f"🔑 **Custom API Keys**\n\nYou have loaded: `{len(keys)}` keys.\nWorking keys: `{valid_keys}`\n\nIf loaded, the bot will strictly use your personal keys first."
        markup.add(types.InlineKeyboardButton("➕ Load / Refresh Keys", callback_data="set_menu_apikey_input"))
        markup.add(types.InlineKeyboardButton("🔙 Back", callback_data="set_menu_main"))

    elif menu == "long_tts":
        markup = types.InlineKeyboardMarkup(row_width=2)
        sizes = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
        btns = [types.InlineKeyboardButton(f"{'✅ ' if chunk_sz == s else ''}{s} ({s/1000.0}m)", callback_data=f"set_tts_sz_{s}") for s in sizes]
        markup.add(*btns)
        markup.row(types.InlineKeyboardButton("🔙 Back", callback_data="set_menu_main"))
        text = "📏 **Long Voiceover Chunk Size**\nSelect how many characters per chunk to split massive scripts into:"
        
    elif menu == "dataset":
        markup = types.InlineKeyboardMarkup(row_width=2)
        datasets = list(scan_multi_datasets().keys())
        is_mix = "mix" in ds_selected or not ds_selected
        btns = [types.InlineKeyboardButton("✅ Mix (All)" if is_mix else "🔀 Mix (All)", callback_data="set_ds_mix")]
        for d in datasets: btns.append(types.InlineKeyboardButton(f"{'✅ ' if (d in ds_selected and not is_mix) else '📁 '}{d}", callback_data=f"set_ds_{d}"))
        markup.add(*btns)
        markup.row(types.InlineKeyboardButton("🔙 Back", callback_data="set_menu_main"))
        text = "🎬 **Select Video Dataset(s)**\nClick to select multiple specific folders. 'Mix' uses all folders.\n*(Use `/fetch_dataset <url>` to download new folders)*"

    elif menu == "bgm_dataset":
        markup = types.InlineKeyboardMarkup(row_width=2)
        datasets = list(scan_bgm_datasets().keys())
        is_mix = "mix" in bgm_selected or not bgm_selected
        btns = [types.InlineKeyboardButton("✅ Mix (All)" if is_mix else "🔀 Mix (All)", callback_data="set_bgm_mix")]
        for d in datasets: btns.append(types.InlineKeyboardButton(f"{'✅ ' if (d in bgm_selected and not is_mix) else '🎵 '}{d}", callback_data=f"set_bgm_{d}"))
        markup.add(*btns)
        markup.row(types.InlineKeyboardButton("🔙 Back", callback_data="set_menu_main"))
        text = "🎵 **Select Background Music Dataset(s)**\nClick to select multiple specific audio folders. 'Mix' uses all audio folders."

    elif menu == "voice":
        markup = types.InlineKeyboardMarkup(row_width=2)
        voices = ["Puck", "Kore", "Aoede", "Charon", "Fenrir", "Lyra"]
        btns = [types.InlineKeyboardButton(f"🗣 {v}", callback_data=f"set_voice_{v}") for v in voices]
        markup.add(*btns)
        markup.row(types.InlineKeyboardButton("🔙 Back", callback_data="set_menu_main"))
        text = "🗣 **Select Gemini Voice Model**\nChoose the character for the AI voice generation:"

    if message_id:
        try: bot.edit_message_text(text, chat_id=chat_id, message_id=message_id, reply_markup=markup, parse_mode="Markdown")
        except: pass
    else: bot.send_message(chat_id, text, reply_markup=markup, parse_mode="Markdown")

@bot.message_handler(func=lambda m: m.text == "⚙️ Settings")
def show_settings(message): send_settings_ui(message.chat.id, menu="main")

@bot.callback_query_handler(func=lambda call: call.data == "set_menu_apikey_input")
def ask_api_keys(call):
    msg = bot.send_message(call.message.chat.id, "🔑 Send your Gemini API Keys separated by commas:\n*(Send /cancel to abort)*", parse_mode="Markdown")
    bot.register_next_step_handler(msg, process_api_keys)
    
def process_api_keys(message):
    if message.text == '/cancel':
        send_settings_ui(message.chat.id, menu="apikey")
        return
    keys = [k.strip() for k in message.text.replace('\n', ',').split(',') if k.strip()]
    with state_lock:
        if message.chat.id not in user_settings: user_settings[message.chat.id] = {}
        user_settings[message.chat.id]["api_keys"] = keys
        user_api_managers[message.chat.id] = SmartAPIKeyManager(keys)
    bot.send_message(message.chat.id, f"✅ Successfully loaded {len(keys)} custom API keys for your profile.")
    send_settings_ui(message.chat.id, menu="apikey")
    trigger_save()

@bot.callback_query_handler(func=lambda call: call.data.startswith("set_"))
def handle_settings_callbacks(call):
    chat_id = call.message.chat.id
    with state_lock:
        if chat_id not in user_settings: user_settings[chat_id] = {"bgm_volume": 7, "dataset": ["mix"], "bgm_dataset": ["mix"], "voice": "Puck", "overlay_on": True, "overlay_opacity": 75, "visual_cuts": 0, "sub_pos": "center", "sub_bg": False}

    if call.data == "set_menu_dataset": send_settings_ui(chat_id, call.message.message_id, "dataset")
    elif call.data == "set_menu_bgmdataset": send_settings_ui(chat_id, call.message.message_id, "bgm_dataset")
    elif call.data == "set_menu_voice": send_settings_ui(chat_id, call.message.message_id, "voice")
    elif call.data == "set_menu_volume": send_settings_ui(chat_id, call.message.message_id, "volume")
    elif call.data == "set_menu_overlay": send_settings_ui(chat_id, call.message.message_id, "overlay")
    elif call.data == "set_menu_viscuts": send_settings_ui(chat_id, call.message.message_id, "viscuts")
    elif call.data == "set_menu_apikey": send_settings_ui(chat_id, call.message.message_id, "apikey")
    elif call.data == "set_menu_long_tts": send_settings_ui(chat_id, call.message.message_id, "long_tts")
    elif call.data == "set_menu_main": send_settings_ui(chat_id, call.message.message_id, "main")
    
    elif call.data.startswith("set_vc_"):
        val = int(call.data.split("_")[-1])
        with state_lock: user_settings[chat_id]["visual_cuts"] = val
        send_settings_ui(chat_id, call.message.message_id, "viscuts")
        trigger_save()
        
    elif call.data.startswith("set_tts_sz_"):
        sz = int(call.data.split("_")[-1])
        with state_lock: user_settings[chat_id]["long_tts_size"] = sz
        send_settings_ui(chat_id, call.message.message_id, "long_tts")
        trigger_save()
        
    elif call.data.startswith("set_ds_"):
        ds_name = call.data[7:]
        with state_lock:
            ds_list = user_settings[chat_id].get("dataset", ["mix"])
            if not isinstance(ds_list, list): ds_list = [ds_list]
            if ds_name == "mix": ds_list = ["mix"]
            else:
                if "mix" in ds_list: ds_list.remove("mix")
                if ds_name in ds_list: 
                    ds_list.remove(ds_name)
                    if not ds_list: ds_list = ["mix"]
                else: ds_list.append(ds_name)
            user_settings[chat_id]["dataset"] = ds_list
        send_settings_ui(chat_id, call.message.message_id, "dataset")
        trigger_save()

    elif call.data.startswith("set_bgm_"):
        ds_name = call.data[8:]
        with state_lock:
            ds_list = user_settings[chat_id].get("bgm_dataset", ["mix"])
            if not isinstance(ds_list, list): ds_list = [ds_list]
            if ds_name == "mix": ds_list = ["mix"]
            else:
                if "mix" in ds_list: ds_list.remove("mix")
                if ds_name in ds_list: 
                    ds_list.remove(ds_name)
                    if not ds_list: ds_list = ["mix"]
                else: ds_list.append(ds_name)
            user_settings[chat_id]["bgm_dataset"] = ds_list
        send_settings_ui(chat_id, call.message.message_id, "bgm_dataset")
        trigger_save()
        
    elif call.data.startswith("set_voice_"):
        v_name = call.data[10:]
        with state_lock: user_settings[chat_id]["voice"] = v_name
        send_settings_ui(chat_id, call.message.message_id, "main")
        trigger_save()

@bot.callback_query_handler(func=lambda call: call.data == "toggle_long_tts")
def handle_long_tts_toggle(call):
    chat_id = call.message.chat.id
    with state_lock:
        if chat_id not in user_settings: user_settings[chat_id] = {}
        user_settings[chat_id]["long_tts"] = not user_settings[chat_id].get("long_tts", False)
    send_settings_ui(chat_id, call.message.message_id, "main")
    trigger_save()

@bot.callback_query_handler(func=lambda call: call.data == "toggle_sub_pos")
def handle_sub_pos_toggle(call):
    chat_id = call.message.chat.id
    with state_lock:
        if chat_id not in user_settings: user_settings[chat_id] = {}
        curr = user_settings[chat_id].get("sub_pos", "center")
        user_settings[chat_id]["sub_pos"] = "bottom" if curr == "center" else "center"
    send_settings_ui(chat_id, call.message.message_id, "main")
    trigger_save()

@bot.callback_query_handler(func=lambda call: call.data == "toggle_sub_bg")
def handle_sub_bg_toggle(call):
    chat_id = call.message.chat.id
    with state_lock:
        if chat_id not in user_settings: user_settings[chat_id] = {}
        user_settings[chat_id]["sub_bg"] = not user_settings[chat_id].get("sub_bg", False)
    send_settings_ui(chat_id, call.message.message_id, "main")
    trigger_save()

@bot.callback_query_handler(func=lambda call: call.data == "toggle_overlay")
def handle_overlay_toggle(call):
    chat_id = call.message.chat.id
    with state_lock:
        if chat_id not in user_settings: user_settings[chat_id] = {}
        user_settings[chat_id]["overlay_on"] = not user_settings[chat_id].get("overlay_on", True)
    send_settings_ui(chat_id, call.message.message_id, "overlay")
    trigger_save()

@bot.callback_query_handler(func=lambda call: call.data.startswith("overop_"))
def handle_overlay_opacity(call):
    action = call.data.split("_")[1]
    if action == "none": return
    chat_id = call.message.chat.id
    with state_lock:
        if chat_id not in user_settings: user_settings[chat_id] = {}
        op = user_settings[chat_id].get("overlay_opacity", 75)
        if action == "-5": op = max(0, op - 5)
        elif action == "+5": op = min(100, op + 5)
        user_settings[chat_id]["overlay_opacity"] = op
    send_settings_ui(chat_id, call.message.message_id, "overlay")
    trigger_save()

@bot.callback_query_handler(func=lambda call: call.data.startswith("vol_"))
def handle_volume(call):
    action = call.data.split("_")[1]
    if action == "none": return
    chat_id = call.message.chat.id
    with state_lock:
        if chat_id not in user_settings: user_settings[chat_id] = {"bgm_volume": 7, "dataset": ["mix"], "voice": "Puck"}
        vol = user_settings[chat_id]["bgm_volume"]
        if action == "-10": vol = max(1, vol - 10)
        elif action == "-1": vol = max(1, vol - 1)
        elif action == "+10": vol = min(100, vol + 10)
        elif action == "+1": vol = min(100, vol + 1)
        user_settings[chat_id]["bgm_volume"] = vol
    send_settings_ui(chat_id, call.message.message_id, "volume")
    trigger_save()

@bot.callback_query_handler(func=lambda call: call.data in ["req_cancel_menu", "do_cancel_current", "do_cancel_all", "do_cancel_abort"])
def handle_cancellations(call):
    chat_id = call.message.chat.id
    if call.data == "req_cancel_menu":
        with state_lock: cancel_mode[chat_id] = True
        update_queue_ui(chat_id, force=True)
    elif call.data == "do_cancel_abort":
        with state_lock: cancel_mode[chat_id] = False
        update_queue_ui(chat_id, force=True)
    elif call.data == "do_cancel_current":
        with state_lock:
            cancel_mode[chat_id] = False
            jobs = jobs_by_chat.get(chat_id, [])
            active_job = next((j for j in jobs if j["status"] in ["generating_audio", "scanning", "rendering"]), None)
            if active_job: active_job["cancelled"] = True
        update_queue_ui(chat_id, force=True)
        try: bot.answer_callback_query(call.id, "🛑 Current batch is cancelling...", show_alert=True)
        except: pass
    elif call.data == "do_cancel_all":
        with state_lock:
            cancel_mode[chat_id] = False
            jobs = jobs_by_chat.get(chat_id, [])
            for j in jobs:
                if j["status"] in ["generating_audio", "scanning", "rendering", "queued_scan"]: j["cancelled"] = True
        update_queue_ui(chat_id, force=True)
        try: bot.answer_callback_query(call.id, "💥 Entire queue is cancelling...", show_alert=True)
        except: pass

# ==========================================
# ENDLESS PAGINATED HISTORY MENU
# ==========================================
def get_history_ui(chat_id, page=0):
    with state_lock: history_list = chat_history.get(chat_id, [])
    if not history_list: return "📜 No history found. Please generate some batches first!", None
        
    ist_tz = timezone(timedelta(hours=5, minutes=30))
    now_date = datetime.now(ist_tz).date()
    
    ITEMS_PER_PAGE = 10
    total_pages = max(1, (len(history_list) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)
    if page >= total_pages: page = max(0, total_pages - 1)
    
    start_idx = page * ITEMS_PER_PAGE
    page_items = history_list[start_idx : start_idx + ITEMS_PER_PAGE]
    
    grouped_history = {}
    for idx_offset, h in enumerate(page_items):
        idx = start_idx + idx_offset + 1
        dt = None
        for fmt in ("%d %b %y, %I:%M %p", "%d-%m-%Y %I:%M %p"):
            try:
                dt = datetime.strptime(h.get('time', ''), fmt)
                break
            except ValueError: continue
                
        if dt:
            delta_days = (now_date - dt.date()).days
            time_part = dt.strftime("%I:%M %p")
            day_label = "Today" if delta_days == 0 else "Yesterday" if delta_days == 1 else dt.strftime("%d %b")
        else: day_label, time_part = "Older", ""
            
        short_n = shorten_name(h['name'])
        grouped_history.setdefault(day_label, []).append({'sno': idx, 'name': short_n, 'count': h['count'], 'time': time_part, 'batch_id': h['batch_id']})
        
    text_lines = [f"**📜 History (Page {page+1}/{total_pages})**\n━━━━━━━━━━━━━\n"]
    markup = types.InlineKeyboardMarkup(row_width=5)
    buttons = []
    
    for day_label, items in grouped_history.items():
        text_lines.append(f"**{day_label}**")
        for item in items:
            text_lines.append(f"`{item['sno']}.` {item['name']} ({item['count']} Vids) {item['time']}")
            buttons.append(types.InlineKeyboardButton(str(item['sno']), callback_data=f"fetch_{item['batch_id']}"))
        text_lines.append("")
        
    markup.add(*buttons)
    
    nav_buttons = []
    if page < total_pages - 1: nav_buttons.append(types.InlineKeyboardButton("⬅️ Prev", callback_data=f"hist_page_{page+1}"))
    nav_buttons.append(types.InlineKeyboardButton("✅ Done", callback_data="collapse"))
    if page > 0: nav_buttons.append(types.InlineKeyboardButton("Next ➡️", callback_data=f"hist_page_{page-1}"))
        
    markup.row(*nav_buttons)
    return "\n".join(text_lines), markup

@bot.message_handler(func=lambda m: m.text == "📜 History")
def show_history(message):
    text, markup = get_history_ui(message.chat.id, 0)
    if markup: bot.send_message(message.chat.id, text, parse_mode="Markdown", reply_markup=markup)
    else: bot.reply_to(message, text)

@bot.callback_query_handler(func=lambda call: call.data.startswith("hist_page_"))
def handle_history_pagination(call):
    page = int(call.data.split("_")[2])
    text, markup = get_history_ui(call.message.chat.id, page)
    try: bot.edit_message_text(text, chat_id=call.message.chat.id, message_id=call.message.message_id, parse_mode="Markdown", reply_markup=markup)
    except: pass
    try: bot.answer_callback_query(call.id)
    except: pass

@bot.callback_query_handler(func=lambda call: call.data == "collapse")
def handle_collapse(call):
    try: bot.delete_message(call.message.chat.id, call.message.message_id)
    except: pass

@bot.callback_query_handler(func=lambda call: call.data.startswith("fetch_"))
def handle_fetch(call):
    batch_id = call.data.split("_", 1)[1]
    with state_lock: channel_msg_ids = batch_registry.get(batch_id, [])
    if not channel_msg_ids:
        try: bot.answer_callback_query(call.id, "⏳ No successful videos found!", show_alert=True)
        except: pass
        return
        
    try: bot.answer_callback_query(call.id, f"✅ Fetching {len(channel_msg_ids)} ready videos...")
    except: pass
    collapse_kb = types.InlineKeyboardMarkup().add(types.InlineKeyboardButton("❌ Collapse", callback_data="collapse"))
    for msg_id in channel_msg_ids:
        try: bot.copy_message(chat_id=call.message.chat.id, from_chat_id=CHANNEL_ID, message_id=msg_id, reply_markup=collapse_kb)
        except: pass

# ==========================================
# TEXT BUFFERING & QUEUING ENGINE
# ==========================================
def queue_text_job(chat_id, text):
    task = {
        "type": "text", "text": text,
        "status": "queued_scan", "batch_id": f"b_{int(time.time())}_{random.randint(10, 99)}", 
        "file_path": None, "chat_id": chat_id, "cancelled": False,
        "total_videos": 0, "completed_videos": 0, "failed_videos": 0, "progress_pct": 0,
        "sub_jobs": [], "live_logs": [], "api_fails": 0
    }
    with state_lock:
        chat_lang_counters.setdefault(chat_id, {})
        jobs_by_chat.setdefault(chat_id, []).append(task)
    move_queue_to_bottom(chat_id)
    master_task_queue.put(task)

def flush_text_buffer(chat_id):
    with buffer_lock:
        if chat_id not in text_buffers: return
        text = text_buffers[chat_id]['text']
        del text_buffers[chat_id]
    queue_text_job(chat_id, text)

@bot.message_handler(func=lambda m: m.text and not m.text.startswith('/'))
def handle_text_script(message):
    text = message.text
    if text in ["📜 History", "⚙️ Settings"]: return 
    
    chat_id = message.chat.id
    with buffer_lock:
        if chat_id in text_buffers:
            text_buffers[chat_id]['timer'].cancel()
            text_buffers[chat_id]['text'] += "\n" + text
        else:
            text_buffers[chat_id] = {'text': text}
        
        timer = threading.Timer(2.0, flush_text_buffer, args=[chat_id])
        text_buffers[chat_id]['timer'] = timer
        timer.start()

# ==========================================
# VIDEO RENDERING ENGINE
# ==========================================
def process_video_job(sj, parent_task):
    chat_id = parent_task["chat_id"]
    if parent_task.get("cancelled"):
        with state_lock: parent_task["failed_videos"] += 1
        return
        
    if not sj.get("ass_subtitle_path"):
        with state_lock: parent_task["failed_videos"] += 1
        return

    job_id = sj["job_id"]
    audio_output_path, audio_duration = sj["audio_output_path"], sj["audio_duration"]
    ass_subtitle_path, llm_metadata = sj["ass_subtitle_path"], sj["llm_metadata"]
    
    with state_lock: sj["status_label"] = "🔍 Std"
    update_queue_ui(chat_id)
    
    def update_parent_progress(pct):
        with state_lock: 
            sj["progress_pct"] = pct
            total_vids = max(1, parent_task.get("total_videos", 1))
            parent_task["progress_pct"] = int(sum(j.get("progress_pct", 0) for j in parent_task.get("sub_jobs", [])) / total_vids)
        update_queue_ui(chat_id)

    update_parent_progress(10)
    output_dir = WORKING_DIR
    for f in os.listdir("/tmp"):
        if f.startswith(f"norm_{job_id}_") or f.startswith(f"concat_{job_id}"):
            try: os.remove(os.path.join("/tmp", f))
            except: pass
            
    try:
        if parent_task.get("cancelled"): raise Exception("Cancelled by user")

        with state_lock: 
            user_dataset = user_settings.get(chat_id, {}).get("dataset", ["mix"])
            user_bgm_dataset = user_settings.get(chat_id, {}).get("bgm_dataset", ["mix"])
            overlay_on = user_settings.get(chat_id, {}).get("overlay_on", True)
            overlay_op = user_settings.get(chat_id, {}).get("overlay_opacity", 75)
            vis_cuts = user_settings.get(chat_id, {}).get("visual_cuts", 0)
            
            if not isinstance(user_dataset, list): user_dataset = [user_dataset]
            if not isinstance(user_bgm_dataset, list): user_bgm_dataset = [user_bgm_dataset]
            
        dataset_map = scan_multi_datasets()
        filtered_map = {}
        if "mix" in user_dataset or not user_dataset: filtered_map = dataset_map
        else:
            for d in user_dataset:
                if d in dataset_map: filtered_map[d] = dataset_map[d]
        if not filtered_map: raise Exception(f"No valid video datasets found. Have you downloaded them via /fetch_dataset?")
            
        selected_clips, _ = build_random_background_multi_dataset(filtered_map, audio_duration, vis_cuts)
        
        bgm_dataset_map = scan_bgm_datasets()
        filtered_bgm_map = {}
        if "mix" in user_bgm_dataset or not user_bgm_dataset: filtered_bgm_map = bgm_dataset_map
        else:
            for d in user_bgm_dataset:
                if d in bgm_dataset_map: filtered_bgm_map[d] = bgm_dataset_map[d]
                
        music_pool = []
        for d in filtered_bgm_map: music_pool.extend(filtered_bgm_map[d])

        bgm_path = build_background_music(music_pool, audio_duration, job_id) if music_pool else None
        
        update_parent_progress(30)
        normalized_files = [None] * len(selected_clips)
        
        with ThreadPoolExecutor(max_workers=min(len(selected_clips), 2)) as executor:
            future_to_idx = {executor.submit(standardize_clip_task, idx, clip, job_id, overlay_on, overlay_op): idx for idx, clip in enumerate(selected_clips)}
            completed_norm = 0
            for future in as_completed(future_to_idx):
                if parent_task.get("cancelled"): raise Exception("Cancelled by user")
                idx = future_to_idx[future]
                if res_path := future.result():
                    normalized_files[idx] = res_path
                    completed_norm += 1
                    update_parent_progress(30 + int((completed_norm / len(selected_clips)) * 40))

        if parent_task.get("cancelled"): raise Exception("Cancelled by user")

        normalized_files = [f for f in normalized_files if f is not None]
        if not normalized_files: raise Exception("All video segments corrupted during standardization.")

        with state_lock: sj["status_label"] = "⚙️ Render"
        update_queue_ui(chat_id)

        actual_durs = [get_media_duration(f) for f in normalized_files]
        final_video_path = os.path.join(output_dir, f"final_{job_id}.mp4")
        ffmpeg_cmd = ['ffmpeg', '-y']
        
        for f in normalized_files: ffmpeg_cmd.extend(['-i', f])
            
        ffmpeg_cmd.extend(['-i', audio_output_path])
        tts_idx = len(normalized_files)
        
        if bgm_path:
            ffmpeg_cmd.extend(['-i', bgm_path])
            bgm_idx = tts_idx + 1

        v_inputs = len(normalized_files)
        if v_inputs > 1:
            xfade_filters = []
            last_node = "0:v"
            current_offset = 0.0
            for i in range(1, v_inputs):
                dur_prev = actual_durs[i-1]
                if dur_prev < 0.6: dur_prev = 0.6 
                current_offset += (dur_prev - 0.5)
                out_node = f"v{i}"
                xfade_filters.append(f"[{last_node}][{i}:v]xfade=transition=fade:duration=0.5:offset={current_offset:.3f}[{out_node}]")
                last_node = out_node
            
            v_filter_str = ";".join(xfade_filters)
            final_v_node = f"[{last_node}]"
        else:
            v_filter_str = ""
            final_v_node = "[0:v]"

        safe_ass_path = ass_subtitle_path.replace("\\", "/").replace(":", "\\:")
        if v_filter_str: v_filter_str += f";{final_v_node}subtitles={safe_ass_path}[final_v]"
        else: v_filter_str = f"{final_v_node}subtitles={safe_ass_path}[final_v]"

        with state_lock: bgm_vol_float = user_settings.get(chat_id, {}).get("bgm_volume", 7) / 100.0

        if bgm_path: a_filter_str = f"[{tts_idx}:a]volume=1.0[v_aud];[{bgm_idx}:a]volume={bgm_vol_float:.2f}[bgm_aud];[v_aud][bgm_aud]amix=inputs=2:duration=first:dropout_transition=0[outa]"
        else: a_filter_str = f"[{tts_idx}:a]volume=1.0[outa]"
            
        full_filter_complex = f"{v_filter_str};{a_filter_str}"

        ffmpeg_cmd.extend(['-filter_complex', full_filter_complex])
        ffmpeg_cmd.extend([
            '-map', '[final_v]',
            '-map', '[outa]',
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '26', 
            '-maxrate', '1.2M', '-bufsize', '2.4M', '-profile:v', 'high', '-pix_fmt', 'yuv420p',
            '-c:a', 'aac', '-b:a', '64k', '-t', f'{audio_duration:.2f}', 
            '-progress', 'pipe:1', final_video_path
        ])
        
        update_parent_progress(80)
        process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        last_pipe_update, error_log_tail = time.time(), []
        
        for line in process.stdout:
            if parent_task.get("cancelled"):
                process.terminate()
                raise Exception("Cancelled by user")

            error_log_tail.append(line)
            if len(error_log_tail) > 100: error_log_tail.pop(0)
            if "out_time_us=" in line:
                try:
                    val = line.strip().split('=')[1]
                    if val != "N/A" and val.lstrip('-').isdigit() and audio_duration > 0:
                        render_progress = min(1.0, (int(val) / 1000000.0) / audio_duration)
                        if time.time() - last_pipe_update > 2.0:
                            update_parent_progress(80 + int(render_progress * 19))
                            last_pipe_update = time.time()
                except: pass
                    
        process.wait()
        if process.returncode != 0 and not parent_task.get("cancelled"): raise Exception("FFmpeg compilation failed")
        if parent_task.get("cancelled"): raise Exception("Cancelled by user")
            
        with state_lock: sj["status_label"] = "🚀 Upload"
        update_queue_ui(chat_id)

        caption_text = f"<code>{llm_metadata}</code>"
        channel_msg = None
        for attempt in range(3):
            if parent_task.get("cancelled"): raise Exception("Cancelled by user")
            try:
                with open(final_video_path, 'rb') as video_file: channel_msg = bot.send_video(CHANNEL_ID, video_file, caption=caption_text, parse_mode="HTML", timeout=120)
                break 
            except Exception as e: time.sleep(5)
                
        if not channel_msg: raise Exception("Failed to upload video to Telegram after 3 attempts.")
            
        with state_lock:
            batch_registry.setdefault(parent_task["batch_id"], []).append(channel_msg.message_id)
            parent_task["completed_videos"] += 1
            sj["status_label"] = "✅ OK"

    except Exception as e:
        if str(e) != "Cancelled by user": print(f"❌ Error rendering video: {e}")
        with state_lock: 
            parent_task["failed_videos"] += 1 
            sj["status_label"] = "❌ Fail"
    finally:
        for p in [audio_output_path, final_video_path, ass_subtitle_path, bgm_path]:
            if p and os.path.exists(p): os.remove(p)
        if 'normalized_files' in locals():
            for f in normalized_files:
                if f and os.path.exists(f): os.remove(f)
        update_parent_progress(100)

def global_master_worker():
    global global_active_job
    print("💼 Global Sequential Pipeline Worker Started!")
    while True:
        task = master_task_queue.get()
        global_active_job = task
        chat_id = task["chat_id"]
        try:
            if task.get("cancelled"): raise Exception("Cancelled by user")

            if task.get("type") == "text":
                with state_lock: 
                    task["status"] = "generating_audio"
                    user_voice = user_settings.get(chat_id, {}).get("voice", "Puck")
                    long_tts = user_settings.get(chat_id, {}).get("long_tts", False)
                    chunk_sz = user_settings.get(chat_id, {}).get("long_tts_size", 1500)
                
                add_log(task, chat_id, "🎙️ Init AI Voice...")
                temp_tg_path = f"/tmp/tg_audio_{chat_id}_{int(time.time())}.wav"
                
                if long_tts: chunks = chunk_text(task["text"], chunk_sz)
                else: chunks = [task["text"]]
                    
                def api_fail_cb(key, err):
                    with state_lock: task["api_fails"] = task.get("api_fails", 0) + 1
                    update_queue_ui(chat_id)
                    
                results = [None] * len(chunks)
                add_log(task, chat_id, f"⚡ Gen {len(chunks)} chunks...")
                
                with ThreadPoolExecutor(max_workers=min(5, len(chunks))) as exe:
                    futures = {exe.submit(generate_audio_with_gemini, c, user_voice, "gemini-2.5-flash-preview-tts", chat_id, 15, api_fail_cb): i for i, c in enumerate(chunks)}
                    
                    for f in as_completed(futures):
                        if task.get("cancelled"): raise Exception("Cancelled by user")
                        idx = futures[f]
                        audio_data, status = f.result()
                        
                        if not audio_data: raise Exception(status) 
                            
                        results[idx] = audio_data
                        with state_lock: task["completed_chunks"] = task.get("completed_chunks", 0) + 1
                        add_log(task, chat_id, f"✅ Voice {task['completed_chunks']}/{len(chunks)} OK")

                add_log(task, chat_id, "🔄 Merging chunks...")
                combined_audio = AudioSegment.empty()
                for idx, audio_data in enumerate(results):
                    if task.get("cancelled"): raise Exception("Cancelled by user")
                    temp_chunk_path = f"/tmp/chunk_{chat_id}_{int(time.time())}_{idx}.wav"
                    if not save_audio_to_file(audio_data, temp_chunk_path)[0]: raise Exception("Failed to save chunk audio")
                    combined_audio += AudioSegment.from_wav(temp_chunk_path)
                    os.remove(temp_chunk_path)
                    
                combined_audio.export(temp_tg_path, format="wav")
                task["file_path"] = temp_tg_path

            if task.get("cancelled"): raise Exception("Cancelled by user")

            with state_lock: task["status"] = "scanning"
            add_log(task, chat_id, "🎧 Transcribing...")
            
            result = whisper_model.transcribe(task["file_path"], word_timestamps=True, initial_prompt="bingo bingo", condition_on_previous_text=False, fp16=False)
            full_lang_name = LANGUAGES.get(result.get("language", "en").lower(), "en").capitalize()
            
            add_log(task, chat_id, "✂️ Cutting clips...")
            all_words, cut_regions = [], []
            for segment in result.get("segments", []):
                for w in segment.get("words", []):
                    all_words.append(w)
                    if is_bingo_trigger(w.get("word", "")): cut_regions.append((w["start"], w["end"]))
                        
            merged_regions = []
            for s, e in cut_regions:
                if not merged_regions: merged_regions.append([s, e])
                else:
                    if s - merged_regions[-1][1] <= 3.0: merged_regions[-1][1] = max(merged_regions[-1][1], e)
                    else: merged_regions.append([s, e])
                        
            audio_seg, current_pos_ms, PADDING_MS, valid_regions = AudioSegment.from_file(task["file_path"]), 0, 400, []
            for s, e in merged_regions:
                cut_start, cut_end = max(0, int(s * 1000) - PADDING_MS), min(len(audio_seg), int(e * 1000) + PADDING_MS)
                if cut_start > current_pos_ms: valid_regions.append((current_pos_ms / 1000.0, cut_start / 1000.0))
                current_pos_ms = cut_end
                
            if current_pos_ms < len(audio_seg): valid_regions.append((current_pos_ms / 1000.0, len(audio_seg) / 1000.0))
            if not valid_regions: valid_regions = [(0.0, len(audio_seg) / 1000.0)]
                
            if task.get("cancelled"): raise Exception("Cancelled by user")

            sub_jobs = []
            for i, (vs, ve) in enumerate(valid_regions):
                sub_jobs.append({
                    "job_id": f"{int(time.time())}_{random.randint(1000, 9999)}_{i}",
                    "vid_idx": i+1,
                    "status_label": "⏳ Wait",
                    "progress_pct": 0,
                    "audio_output_path": None, "audio_duration": 0,
                    "ass_subtitle_path": None, "llm_metadata": ""
                })

            with state_lock:
                chat_dict = chat_lang_counters.setdefault(chat_id, {})
                lang_data = chat_dict.get(full_lang_name)
                
                today_str = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%Y-%m-%d")
                if isinstance(lang_data, dict):
                    if lang_data.get("date") == today_str: lang_count = lang_data.get("count", 0) + 1
                    else: lang_count = 1
                else: lang_count = (lang_data or 0) + 1
                    
                chat_lang_counters[chat_id][full_lang_name] = {"date": today_str, "count": lang_count}
                task["display_name"] = f"{shorten_name(full_lang_name)} {lang_count}"
                task["total_videos"] = len(valid_regions)
                task["sub_jobs"] = sub_jobs
                task["status"] = "rendering"
                
                sub_pos = user_settings.get(chat_id, {}).get("sub_pos", "center")
                sub_bg = user_settings.get(chat_id, {}).get("sub_bg", False)
                
            update_queue_ui(chat_id, force=True)
                
            with ThreadPoolExecutor(max_workers=10) as exe:
                futures = [exe.submit(prep_chunk_parallel, sj, vs, ve, all_words, audio_seg, WORKING_DIR, chat_id, sub_pos, sub_bg) for sj, (vs, ve) in zip(sub_jobs, valid_regions)]
                for f in as_completed(futures): f.result()
                    
            with ThreadPoolExecutor(max_workers=2) as render_exe:
                render_futures = [render_exe.submit(process_video_job, sj, task) for sj in sub_jobs]
                for f in as_completed(render_futures): f.result()

            if task.get("cancelled"): raise Exception("Cancelled by user")
                    
            with state_lock:
                task["status"] = "completed"
                task["progress_pct"] = 100
                c_h = chat_history.setdefault(chat_id, [])
                c_h.insert(0, {"batch_id": task["batch_id"], "name": task["display_name"], "count": task["total_videos"], "time": datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%d %b %y, %I:%M %p")})
                chat_history[chat_id] = clean_old_history(c_h) 
                
            trigger_save() 

        except Exception as e:
            err_msg = str(e)
            if err_msg == "Cancelled by user":
                print(f"🛑 Batch Cancelled: {task['batch_id']}")
                with state_lock: task["status"] = "cancelled"
            elif err_msg == "API_FAIL_USER":
                with state_lock: task["status"] = "failed_api_user"
            elif err_msg == "API_FAIL_DEFAULT":
                with state_lock: task["status"] = "failed_api_default"
            else:
                print(f"Global Worker Error: {e}")
                with state_lock: task["status"] = "failed_master"
        finally:
            global_active_job = None
            if task.get("file_path") and os.path.exists(task["file_path"]): os.remove(task["file_path"])
            master_task_queue.task_done()
            
            with state_lock:
                pending_jobs = [j for j in jobs_by_chat.get(chat_id, []) if j["status"] in ["queued_scan", "generating_audio", "scanning", "rendering"]]
            
            if not pending_jobs: move_queue_to_bottom(chat_id)
            else: update_queue_ui(chat_id, force=True)
            
            if not master_task_queue.empty():
                next_task = master_task_queue.queue[0]
                update_queue_ui(next_task["chat_id"], force=True)

threading.Thread(target=global_master_worker, daemon=True).start()

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "👋 Welcome to the Auto-Video Maker!\n\n"
                          "⚙️ **Kaggle Setup via Chat:**\n"
                          "1. `/set_kaggle <username> <api_key>` - Set Kaggle Login.\n"
                          "2. `/fetch_dataset <url-or-slug>` - Download video datasets.\n\n"
                          "🎙️ Upload Voiceover files OR Send me a Text Script (with 'bingo bingo') to generate AI Voice!\n\n"
                          "⚡ Mode: `Pure Cinematic Glow + 1080p Smart VBV Limit`", reply_markup=get_main_keyboard(), parse_mode="Markdown")

@bot.message_handler(content_types=['audio', 'voice', 'document'])
def handle_incoming_uploads(message):
    if message.document and message.document.file_name.lower().endswith('.txt'):
        file_info = bot.get_file(message.document.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        try:
            text = downloaded_file.decode('utf-8')
            queue_text_job(message.chat.id, text)
        except:
            bot.send_message(message.chat.id, "❌ Failed to read text file. Please ensure it is saved with UTF-8 encoding.")
        return

    file_id = None
    if message.voice: file_id = message.voice.file_id
    elif message.audio: file_id = message.audio.file_id
    elif message.document and message.document.file_name.lower().endswith(('.mp3', '.wav', '.ogg', '.m4a', '.aac', '.mpga', '.opus')): file_id = message.document.file_id
    if not file_id: return
        
    file_info = bot.get_file(file_id)
    ext = os.path.splitext(file_info.file_path)[1]
    if not ext: ext = ".ogg" if message.voice else ".mp3"
        
    temp_tg_path = f"/tmp/tg_audio_{message.chat.id}_{int(time.time())}{ext}"
    task = {
        "type": "audio", "status": "queued_scan", "batch_id": f"b_{int(time.time())}_{random.randint(10, 99)}", 
        "file_path": temp_tg_path, "chat_id": message.chat.id, "cancelled": False,
        "total_videos": 0, "completed_videos": 0, "failed_videos": 0, "progress_pct": 0, "sub_jobs": [], "live_logs": [], "api_fails": 0
    }
    try:
        downloaded_file = bot.download_file(file_info.file_path)
        with open(temp_tg_path, 'wb') as f: f.write(downloaded_file)
        with state_lock:
            chat_lang_counters.setdefault(message.chat.id, {})
            jobs_by_chat.setdefault(message.chat.id, []).append(task)
        move_queue_to_bottom(message.chat.id)
        master_task_queue.put(task)
    except Exception as err:
        bot.send_message(message.chat.id, f"❌ Failed starting download: {err}")
        if os.path.exists(temp_tg_path): os.remove(temp_tg_path)

if __name__ == "__main__":
    print("🤖 Massive Multi-Threading Sequential Batch Video Bot is running on GitHub Actions!")
    try:
        load_history()
        bot.remove_webhook()
        time.sleep(1)
    except: pass
    bot.infinity_polling(skip_pending=True)

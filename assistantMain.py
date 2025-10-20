# assistantMain.py
# SAINT Desktop Agent — with YouTube search + pico-llm (gemma-7b-it-403) support
#
# Required / optional packages (best-effort):
# pip install flask requests pyttsx3 pyautogui pillow spotipy SpeechRecognition pyaudio pvporcupine pv_rhino
# For pico-llm: install whatever package you use to load .pllm models (example: pico-llm). This script
# attempts multiple common import names but you must have your model loader installed.
#
# On Windows, if pyaudio fails: pip install pipwin && pipwin install pyaudio

import os, sys, json, time, threading, subprocess, webbrowser, re, traceback, struct
from dataclasses import dataclass
from pathlib import Path
from io import BytesIO

# GUI
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox, scrolledtext

# Networking / Flask
import requests
from flask import Flask, request, jsonify, redirect, url_for

# Optional extras
try:
    import pyttsx3
    HAS_TTS = True
except Exception:
    pyttsx3 = None
    HAS_TTS = False

try:
    import pyautogui
    from PIL import Image
    HAS_PYAUTOGUI = True
except Exception:
    HAS_PYAUTOGUI = False

try:
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth
    HAS_SPOTIPY = True
except Exception:
    spotipy = None
    SpotifyOAuth = None
    HAS_SPOTIPY = False

try:
    import speech_recognition as sr
    HAS_SR = True
except Exception:
    sr = None
    HAS_SR = False

# Picovoice optional imports
PICOVOICE_AVAILABLE = False
PV_PORCUPINE = None
PV_RHINO = None
PAUDIO = None
try:
    import pvporcupine as PV_PORCUPINE
    import pv_rhino as PV_RHINO
    import pyaudio as PAUDIO
    PICOVOICE_AVAILABLE = True
except Exception:
    try:
        import picovoice  # unified package fallback (not used directly here)
        PICOVOICE_AVAILABLE = True
    except Exception:
        PICOVOICE_AVAILABLE = False

# ----------------------------
# Pico-LLM integration (best-effort)
# Tries to import common packages and load a local .pllm model.
# If no compatible loader found, falls back to rule-based responses.
# ----------------------------
PICO_AVAILABLE = False
PICO = None
PICO_MODEL = None
PICO_MODEL_PATH = os.environ.get("PICO_MODEL_PATH") or os.path.join("models", "gemma-7b-it-403.pllm")

# Attempt multiple import names / APIs
try:
    import pico_llm as PICO  # hypothetical
    PICO_AVAILABLE = True
except Exception:
    try:
        import pico as PICO  # alternative
        PICO_AVAILABLE = True
    except Exception:
        try:
            # llama.cpp python bindings as a fallback (if user has it)
            import llama_cpp as PICO
            PICO_AVAILABLE = True
        except Exception:
            PICO = None
            PICO_AVAILABLE = False

def load_pico_model(path):
    """
    Best-effort loader for local .pllm model files. The exact API depends on user's installed package.
    Returns a model object (opaque) or None.
    """
    global PICO, PICO_MODEL
    if not PICO_AVAILABLE:
        return None
    if not path or not os.path.exists(path):
        print("[LLM] Model path not found:", path)
        return None
    try:
        # Try several likely APIs
        # 1) pico_llm.Model(path)
        if hasattr(PICO, "Model"):
            try:
                PICO_MODEL = PICO.Model(path)
                print("[LLM] Loaded model via PICO.Model()")
                return PICO_MODEL
            except Exception:
                pass
        # 2) pico.load_model(path)
        if hasattr(PICO, "load_model"):
            try:
                PICO_MODEL = PICO.load_model(path)
                print("[LLM] Loaded model via PICO.load_model()")
                return PICO_MODEL
            except Exception:
                pass
        # 3) llama_cpp.Llama(model_path=path)
        if hasattr(PICO, "Llama"):
            try:
                PICO_MODEL = PICO.Llama(model_path=path)
                print("[LLM] Loaded model via Llama()")
                return PICO_MODEL
            except Exception:
                pass
        # 4) pico.PicoModel / pico.Client
        if hasattr(PICO, "PicoModel"):
            try:
                PICO_MODEL = PICO.PicoModel(path)
                print("[LLM] Loaded model via PicoModel()")
                return PICO_MODEL
            except Exception:
                pass
        # nothing matched
        print("[LLM] Could not find supported API on imported package. Model not loaded.")
    except Exception as e:
        print("[LLM] load_pico_model error:", e)
    return None

def pico_generate(prompt, max_tokens=256, temperature=0.7):
    """
    Best-effort wrapper to generate text using loaded PICO_MODEL.
    If not available or call fails, returns None.
    """
    global PICO, PICO_MODEL
    if PICO_MODEL is None:
        return None
    try:
        # llama_cpp
        if hasattr(PICO_MODEL, "generate") and callable(PICO_MODEL.generate):
            # some llama_cpp APIs use generate
            out = PICO_MODEL.generate(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
            if isinstance(out, dict):
                # llama-cpp return structure may contain 'choices'
                choices = out.get("choices")
                if choices and isinstance(choices, list) and len(choices) > 0:
                    text = choices[0].get("text") or choices[0].get("content") or ""
                    return text
            if isinstance(out, str):
                return out
        # llama_cpp Llama __call__ returns dict with choices
        if hasattr(PICO_MODEL, "completion") and callable(PICO_MODEL.completion):
            try:
                res = PICO_MODEL.completion(prompt, max_tokens=max_tokens)
                if isinstance(res, dict) and "text" in res:
                    return res["text"]
            except Exception:
                pass
        # pico-like generate/predict
        if hasattr(PICO_MODEL, "predict") and callable(PICO_MODEL.predict):
            try:
                return PICO_MODEL.predict(prompt)
            except Exception:
                pass
        if hasattr(PICO_MODEL, "infer") and callable(PICO_MODEL.infer):
            return PICO_MODEL.infer(prompt)
    except Exception as e:
        print("[LLM] generate error:", e)
    return None

# Try to load model at startup (best-effort)
if PICO_AVAILABLE:
    try:
        load_pico_model(PICO_MODEL_PATH)
    except Exception as e:
        print("[LLM] startup load error:", e)

# ----------------------------
# Files & config
# ----------------------------
CAL_FILE = 'calibration.json'
CONFIG_FILE = 'robot_config.json'
HISTORY_FILE = 'conversation_history.json'
SPOTIFY_CACHE = 'spotify_token.json'
PROFILE_FILE = 'profile.json'
ACTIONS_FILE = 'actions.json'
DEFAULT_CONFIG = {'input_mode': 'voice', 'output_mode': 'both'}
DEFAULT_CAL = {
    "NH": {"channel": 0, "min": 200, "max": 584, "zero_deg": 90},
    "NV": {"channel": 1, "min": 230, "max": 520, "zero_deg": 90},
    "EV": {"channel": 2, "min": 150, "max": 430, "zero_deg": 90},
    "EH": {"channel": 3, "min": 160, "max": 320, "zero_deg": 90},
    "M":  {"channel": 4, "min": 160, "max": 300, "zero_deg": 90}
}
EYE_CHANNELS = {'R1':5,'G1':6,'B1':7,'R2':8,'G2':9,'B2':10}

@dataclass
class ServoCal:
    name: str
    channel: int
    min: int
    max: int
    zero_deg: int = 90

# ----------------------------
# ACTION LIST (will be saved to actions.json)
# ----------------------------
ACTION_DEFINITIONS = {
    "open": {"desc": "Open an application, file, or URL (e.g., 'open spotify', 'open command prompt')."},
    "search": {"desc": "Search the web for a query (opens browser)."},
    "youtube_search": {"desc": "Search YouTube for a query (opens results page)."},
    "open_file_vscode": {"desc":"Open a file in VSCode."},
    "open_vscode": {"desc":"Open Visual Studio Code."},
    "type": {"desc":"Simulate typing text."},
    "press": {"desc":"Simulate pressing a key."},
    "click": {"desc":"Simulate a click at coordinates."},
    "tts": {"desc":"Speak text via TTS."},
    "eye": {"desc":"Set eye LED color."},
    "servo_move": {"desc":"Move servo to degree."},
    "run_shell": {"desc":"Run allowed shell commands (very restricted)."},
    "list_bookmarks": {"desc":"List browser bookmarks found locally."},
    "open_bookmark": {"desc":"Open a bookmark by title."},
    "spotify_play": {"desc":"Search and play a song on Spotify."},
    "spotify_pause": {"desc":"Pause Spotify playback."},
    "spotify_resume": {"desc":"Resume Spotify."},
    "spotify_next": {"desc":"Skip to next track."},
    "spotify_previous": {"desc":"Previous track."},
    "spotify_seek": {"desc":"Seek playback position."}
}

# Write actions.json at startup
try:
    with open(ACTIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(ACTION_DEFINITIONS, f, indent=2)
except Exception:
    pass

# ----------------------------
# RobotHead (persona, history, learning)
# ----------------------------
class RobotHead:
    def __init__(self):
        self.cal = self._load_cal()
        self.input_mode = DEFAULT_CONFIG['input_mode']
        self.output_mode = DEFAULT_CONFIG['output_mode']
        self.history = [{'role':'system','content':self._system_context()}]
        self.load_config(); self.load_history()
        self.profile = self._load_profile()
        self.is_speaking = False
        self._tts_lock = threading.Lock()
        self._tts_engine = None
        if HAS_TTS:
            try:
                self._tts_engine = pyttsx3.init()
            except Exception:
                self._tts_engine = None

    def _system_context(self):
        return (
            "You are SAINT, the user's local assistant running on their machine. "
            "Behavior: direct, practical, concise. Don't sugar-coat. "
            "Capabilities: open local apps, control Spotify (with OAuth), search the web, open YouTube search results, read bookmarks, simulate keyboard/mouse, and run a small set of safe shell commands. "
            "Personality: helpful, slightly blunt, friendly; remember user's simple preferences (name, common apps). "
            "Learning: store repeated user preferences in a local profile file and use them to personalize replies. "
            "When performing an action, say a short confirmation: e.g. 'Opening File Explorer.' If an action fails, explain why and provide a fallback."
        )

    # config/history helpers
    def _load_cal(self):
        if os.path.exists(CAL_FILE):
            try:
                with open(CAL_FILE,'r') as f:
                    data = json.load(f)
                for k,v in DEFAULT_CAL.items():
                    if k not in data:
                        data[k] = v
                return {name: ServoCal(name=name, channel=v['channel'], min=v['min'], max=v['max'], zero_deg=v.get('zero_deg',90)) for name,v in data.items()}
            except Exception:
                pass
        return {name: ServoCal(name=name, channel=v['channel'], min=v['min'], max=v['max'], zero_deg=v.get('zero_deg',90)) for name,v in DEFAULT_CAL.items()}

    def save_cal(self):
        out = {name: {'channel':s.channel, 'min':s.min, 'max':s.max, 'zero_deg':s.zero_deg} for name,s in self.cal.items()}
        try:
            with open(CAL_FILE,'w') as f:
                json.dump(out,f,indent=2)
            return True
        except Exception:
            return False

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE,'r') as f:
                    c = json.load(f)
                self.input_mode = c.get('input_mode', self.input_mode)
                self.output_mode = c.get('output_mode', self.output_mode)
            except Exception:
                pass

    def save_config(self):
        try:
            with open(CONFIG_FILE,'w') as f:
                json.dump({'input_mode': self.input_mode, 'output_mode': self.output_mode}, f, indent=2)
            return True
        except Exception:
            return False

    def load_history(self):
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE,'r', encoding='utf-8') as f:
                    h = json.load(f)
                if isinstance(h, list) and len(h) > 0:
                    self.history = h
            except Exception:
                pass

    def save_history(self):
        try:
            with open(HISTORY_FILE,'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2)
            return True
        except Exception:
            return False

    def _load_profile(self):
        default = {"name": None, "action_counts": {}}
        if os.path.exists(PROFILE_FILE):
            try:
                with open(PROFILE_FILE, 'r', encoding='utf-8') as fh:
                    p = json.load(fh)
                if isinstance(p, dict):
                    return p
            except Exception:
                pass
        return default

    def _save_profile(self):
        try:
            with open(PROFILE_FILE, 'w', encoding='utf-8') as fh:
                json.dump(self.profile, fh, indent=2)
        except Exception:
            pass

    def record_action(self, action_name):
        if not action_name:
            return
        ac = self.profile.get("action_counts", {})
        ac[action_name] = ac.get(action_name, 0) + 1
        self.profile["action_counts"] = ac
        self._save_profile()

    def try_extract_and_save_name(self, text):
        if not text:
            return None
        m = re.search(r"\b(?:my name is|i am|i'm|call me)\s+([A-Z][a-zA-Z'-]{1,30})\b", text, flags=re.I)
        if m:
            name = m.group(1).strip()
            self.profile['name'] = name
            self._save_profile()
            return name
        return None

    # hardware placeholders
    def set_eye_color(self, r, g, b):
        print(f"[EYE] set {r},{g},{b}")
        return {"ok": True}

    def move_servo_deg(self, servo_name, deg):
        print(f"[SERVO] move {servo_name} -> {deg}")
        return {"ok": True}

    def run_shell(self, cmd, timeout=30, shell=False):
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, shell=shell)
            return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
        except Exception as e:
            return {"error": str(e)}

    def open_app(self, path_or_cmd):
        if not path_or_cmd:
            return {"error":"no target provided"}
        targ = str(path_or_cmd).strip()
        if targ.startswith("http://") or targ.startswith("https://"):
            webbrowser.open(targ)
            return {"opened": targ}
        tl = targ.lower()
        if "spotify" in tl:
            try:
                if sys.platform.startswith("win"):
                    try:
                        os.startfile("spotify:")
                        return {"opened": "spotify_app"}
                    except Exception:
                        subprocess.Popen('start spotify', shell=True)
                        return {"opened_shell": "start spotify"}
                else:
                    try:
                        subprocess.Popen(["spotify"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        return {"opened": "spotify_app"}
                    except Exception:
                        webbrowser.open("https://open.spotify.com")
                        return {"opened": "https://open.spotify.com"}
            except Exception as e:
                return {"error": str(e)}
        if tl in ("command prompt", "cmd", "terminal", "command"):
            if sys.platform.startswith("win"):
                try:
                    subprocess.Popen('start cmd', shell=True)
                    return {"opened_shell": "start cmd"}
                except Exception as e:
                    return {"error": str(e)}
            elif sys.platform.startswith("darwin"):
                try:
                    subprocess.Popen(["open", "-a", "Terminal"])
                    return {"opened": "Terminal"}
                except Exception as e:
                    return {"error": str(e)}
            else:
                try:
                    subprocess.Popen(["x-terminal-emulator"], shell=False)
                    return {"opened": "terminal"}
                except Exception:
                    try:
                        subprocess.Popen(["gnome-terminal"], shell=False)
                        return {"opened": "gnome-terminal"}
                    except Exception as e:
                        return {"error": str(e)}
        if tl in ("vs", "vscode", "visual studio code", "visual studio"):
            try:
                subprocess.Popen("code", shell=True)
                return {"opened_shell": "code"}
            except Exception:
                if sys.platform.startswith("win"):
                    try:
                        subprocess.Popen(['devenv'], shell=True)
                        return {"opened_shell": "devenv"}
                    except Exception:
                        return {"error": "Could not find VS or code in PATH"}
                else:
                    return {"error": "VS/Code not in PATH"}
        if tl in ("notes", "notepad", "sticky notes"):
            if sys.platform.startswith("win"):
                try:
                    subprocess.Popen('start notepad', shell=True)
                    return {"opened_shell": "notepad"}
                except Exception as e:
                    return {"error": str(e)}
            elif sys.platform.startswith("darwin"):
                try:
                    subprocess.Popen(["open", "-a", "Notes"])
                    return {"opened": "Notes"}
                except Exception as e:
                    return {"error": str(e)}
            else:
                try:
                    subprocess.Popen(["gedit"], shell=False)
                    return {"opened": "gedit"}
                except Exception:
                    return {"error": "no default notes app found"}
        try:
            if os.path.exists(targ):
                if sys.platform.startswith("win"):
                    os.startfile(os.path.abspath(targ))
                    return {"opened": os.path.abspath(targ)}
                else:
                    subprocess.Popen(["xdg-open", targ], shell=False)
                    return {"opened": targ}
            if sys.platform.startswith("win"):
                subprocess.Popen(targ, shell=True)
                return {"opened_shell": targ}
            else:
                subprocess.Popen(targ, shell=True)
                return {"opened_shell": targ}
        except Exception as e:
            return {"error": str(e)}

    def web_search(self, query):
        url = "https://www.google.com/search?q=" + requests.utils.quote(query)
        webbrowser.open(url)
        return {"search_url": url}

    def type_text(self, text, interval=0.03):
        if not HAS_PYAUTOGUI:
            return {"error": "pyautogui not available"}
        try:
            pyautogui.write(text, interval=interval)
            return {"typed": text}
        except Exception as e:
            return {"error": str(e)}

    def press_key(self, key):
        if not HAS_PYAUTOGUI:
            return {"error": "pyautogui not available"}
        try:
            pyautogui.press(key)
            return {"pressed": key}
        except Exception as e:
            return {"error": str(e)}

    # TTS control + immediate stop
    def tts_say(self, text):
        if not HAS_TTS or self._tts_engine is None:
            return {"error": "no tts available"}
        def _speak():
            try:
                with self._tts_lock:
                    self.is_speaking = True
                    self._tts_engine.say(text)
                    self._tts_engine.runAndWait()
            except Exception as e:
                print("[TTS] speak error:", e)
            finally:
                self.is_speaking = False
        # run TTS in background to allow interruption
        t = threading.Thread(target=_speak, daemon=True)
        t.start()
        return {"ok": True}

    def tts_stop(self):
        """
        Immediately stop TTS if running (pyttsx3 supports stop()).
        """
        if not HAS_TTS or self._tts_engine is None:
            self.is_speaking = False
            return {"ok": False, "note":"tts not available"}
        try:
            with self._tts_lock:
                try:
                    self._tts_engine.stop()
                except Exception:
                    pass
                self.is_speaking = False
            return {"ok": True}
        except Exception as e:
            return {"error": str(e)}

    # Use pico-llm if available for chat, otherwise fallback
    def chat_single_answer(self, prompt):
        if not prompt:
            return "Say something and I'll try to help."
        self.history.append({'role': 'user', 'content': prompt})
        self.save_history()
        # attempt name extraction
        name = self.try_extract_and_save_name(prompt)
        # If pico LLM available, use it for a richer response
        if PICO_MODEL is not None:
            # Build a prompt that includes personality and conversation history
            system = self.history[0]['content']
            recent = "\n".join([f"{h['role']}: {h['content']}" for h in self.history[-10:]])
            full_prompt = (f"{system}\n\nUser: {prompt}\n\nRecent conversation:\n{recent}\n\n"
                           f"Respond concisely, directly, and include a one-line confirmation if you performed an action. "
                           f"Personalize if you know the user's name ({self.profile.get('name')}).")
            text = pico_generate(full_prompt, max_tokens=256, temperature=0.75)
            if text:
                assistant_text = text.strip()
                self.history.append({'role':'assistant','content':assistant_text})
                self.save_history()
                return assistant_text
            # else fall through to rule-based

        # Rule-based fallback behavior (keeps previous personality)
        p = prompt.strip()
        pl = p.lower()
        if name:
            assistant_text = f"Nice to meet you, {name}. What can I do for you?"
        elif any(q in pl for q in ["who are you", "what are you", "what is your name"]):
            un = self.profile.get("name") or "friend"
            assistant_text = (f"I'm SAINT — your local assistant. I run on your machine and help with apps, "
                              f"Spotify, files, and small automations. I'll call you {un}.")
        elif any(g in pl for g in ["hello", "hi", "hey"]):
            un = self.profile.get("name") or "there"
            assistant_text = f"Hey {un} — tell me what to open or ask me to do something."
        elif pl.startswith("open ") or pl.startswith("launch ") or pl.startswith("start "):
            action_res = nl_execute_from_text(p)
            assistant_text = action_res.get("assistant") or action_res.get("response") or "Tried to run that."
            if action_res.get("action"):
                self.record_action(action_res.get("action"))
        else:
            assistant_text = f"I heard: \"{p}\". I can open apps, control Spotify, search the web, or run safe commands."
        self.history.append({'role':'assistant','content':assistant_text})
        self.save_history()
        return assistant_text

# ----------------------------
# Spotify controller (minimal)
# ----------------------------
class SpotifyController:
    def __init__(self, cache_path=SPOTIFY_CACHE):
        self.cache_path = cache_path
        self.oauth = None
        self._last_error = None
        self._setup_oauth()
    def _setup_oauth(self):
        if not HAS_SPOTIPY:
            self._last_error = "spotipy not installed"
            self.oauth = None
            return
        client_id = os.environ.get("SPOTIPY_CLIENT_ID") or "0650b3589b5c45269f7f9684e3086482"
        client_secret = os.environ.get("SPOTIPY_CLIENT_SECRET") or "c80df8aa995c477bb58889e2eb8d258"
        redirect_env = os.environ.get("SPOTIPY_REDIRECT_URI") or "http://127.0.0.1:8765/spotify/callback"
        port = os.environ.get("SAINT_PORT", "8765")
        redirect_default = f"http://127.0.0.1:{port}/spotify/callback"
        redirect = redirect_env or redirect_default
        if not client_id or not client_secret:
            self._last_error = "SPOTIPY_CLIENT_ID or SPOTIPY_CLIENT_SECRET not set in env"
            self.oauth = None
            return
        scopes = "user-read-playback-state user-modify-playback-state user-read-currently-playing user-read-private"
        try:
            self.oauth = SpotifyOAuth(client_id=client_id,
                                      client_secret=client_secret,
                                      redirect_uri=redirect,
                                      scope=scopes,
                                      cache_path=self.cache_path)
            self._last_error = None
        except Exception as e:
            self.oauth = None
            self._last_error = f"exception creating SpotifyOAuth: {e}"
    def is_configured(self):
        return HAS_SPOTIPY and (self.oauth is not None)
    def get_auth_url(self):
        if not self.is_configured():
            return None
        try:
            return self.oauth.get_authorize_url()
        except Exception as e:
            self._last_error = f"get_authorize_url error: {e}"
            return None
    def authorize_callback(self, request_args):
        if not self.is_configured():
            return {"error":"spotipy not configured", "diag": self.diagnostics()}
        err = request_args.get('error')
        if err:
            return {"error": err}
        code = request_args.get('code')
        if not code:
            return {"error":"no code in callback"}
        try:
            try:
                token_info = self.oauth.get_access_token(code)
            except TypeError:
                token_info = self.oauth.get_access_token(code, as_dict=True)
            return {"ok": True, "token_info": token_info}
        except Exception as e:
            self._last_error = f"authorize_callback error: {e}"
            return {"error": str(e), "diag": self.diagnostics()}
    def _get_token(self):
        if not self.is_configured():
            return None
        try:
            if hasattr(self.oauth, "get_cached_token"):
                token_info = self.oauth.get_cached_token()
                if token_info and isinstance(token_info, dict):
                    return token_info.get("access_token")
            if self.cache_path and os.path.exists(self.cache_path):
                with open(self.cache_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict):
                    if "access_token" in data:
                        return data.get("access_token")
                    if "token_info" in data and isinstance(data["token_info"], dict):
                        return data["token_info"].get("access_token")
                    if "expires_at" in data or "refresh_token" in data:
                        return data.get("access_token")
            return None
        except Exception as e:
            self._last_error = f"_get_token exception: {e}"
            return None
    def diagnostics(self):
        return {"has_spotipy": HAS_SPOTIPY, "oauth_present": self.oauth is not None, "last_error": self._last_error, "cache_path": self.cache_path}
    def _auth_headers(self):
        token = self._get_token()
        if not token:
            return None
        return {"Authorization": f"Bearer {token}", "Content-Type":"application/json"}
    def current_playback(self):
        headers = self._auth_headers()
        if not headers:
            return {"error":"not authenticated", "diag": self.diagnostics()}
        try:
            r = requests.get("https://api.spotify.com/v1/me/player", headers=headers, timeout=5)
            return r.json()
        except Exception as e:
            return {"error": str(e)}
    def play_uri(self, uri=None, device_id=None, position_ms=0):
        headers = self._auth_headers()
        if not headers:
            return {"error":"not authenticated", "diag": self.diagnostics()}
        body = {}
        if uri:
            if uri.startswith("spotify:track:") or uri.startswith("http"):
                if uri.startswith("http"):
                    m = re.search(r"/track/([A-Za-z0-9]+)", uri)
                    if m:
                        track_id = m.group(1)
                        body["uris"] = [f"spotify:track:{track_id}"]
                else:
                    body["uris"] = [uri]
            else:
                body["context_uri"] = uri
        endpoint = "https://api.spotify.com/v1/me/player/play"
        params = {}
        if device_id:
            params["device_id"] = device_id
        try:
            r = requests.put(endpoint, headers=headers, params=params, data=json.dumps(body), timeout=5)
            if r.status_code in (204,202):
                return {"ok": True}
            if r.status_code == 404:
                return {"error":"no active device", "status_code": r.status_code, "text": r.text, "diag": self.diagnostics()}
            return {"status_code": r.status_code, "text": r.text}
        except Exception as e:
            return {"error": str(e)}
    def pause(self):
        headers = self._auth_headers()
        if not headers:
            return {"error":"not authenticated", "diag": self.diagnostics()}
        try:
            r = requests.put("https://api.spotify.com/v1/me/player/pause", headers=headers, timeout=5)
            if r.status_code in (204,202):
                return {"ok": True}
            if r.status_code == 404:
                return {"error":"no active device", "status_code": r.status_code, "text": r.text, "diag": self.diagnostics()}
            return {"status_code": r.status_code, "text": r.text}
        except Exception as e:
            return {"error": str(e)}
    def resume(self):
        return self.play_uri(None)
    def next_track(self):
        headers = self._auth_headers()
        if not headers:
            return {"error":"not authenticated", "diag": self.diagnostics()}
        try:
            r = requests.post("https://api.spotify.com/v1/me/player/next", headers=headers, timeout=5)
            if r.status_code in (204,202):
                return {"ok": True}
            if r.status_code == 404:
                return {"error":"no active device", "status_code": r.status_code, "text": r.text, "diag": self.diagnostics()}
            return {"status_code": r.status_code, "text": r.text}
        except Exception as e:
            return {"error": str(e)}
    def previous_track(self):
        headers = self._auth_headers()
        if not headers:
            return {"error":"not authenticated", "diag": self.diagnostics()}
        try:
            r = requests.post("https://api.spotify.com/v1/me/player/previous", headers=headers, timeout=5)
            if r.status_code in (204,202):
                return {"ok": True}
            if r.status_code == 404:
                return {"error":"no active device", "status_code": r.status_code, "text": r.text, "diag": self.diagnostics()}
            return {"status_code": r.status_code, "text": r.text}
        except Exception as e:
            return {"error": str(e)}
    def seek(self, position_ms):
        headers = self._auth_headers()
        if not headers:
            return {"error":"not authenticated", "diag": self.diagnostics()}
        try:
            r = requests.put(f"https://api.spotify.com/v1/me/player/seek?position_ms={int(position_ms)}", headers=headers, timeout=5)
            if r.status_code in (204,202):
                return {"ok": True}
            if r.status_code == 404:
                return {"error":"no active device", "status_code": r.status_code, "text": r.text, "diag": self.diagnostics()}
            return {"status_code": r.status_code, "text": r.text}
        except Exception as e:
            return {"error": str(e)}
    def search_and_play(self, query, market="US"):
        headers = self._auth_headers()
        if not headers:
            return {"error":"not authenticated", "diag": self.diagnostics()}
        try:
            q = requests.utils.quote(query)
            r = requests.get(f"https://api.spotify.com/v1/search?q={q}&type=track,album,playlist,artist&limit=5&market={market}", headers=headers, timeout=5)
            if r.status_code != 200:
                return {"status_code": r.status_code, "text": r.text}
            data = r.json()
            tracks = data.get("tracks", {}).get("items", [])
            if tracks:
                track = tracks[0]
                uri = track.get("uri")
                return self.play_uri(uri)
            playlists = data.get("playlists", {}).get("items", [])
            if playlists:
                ctx_uri = playlists[0].get("uri")
                return self.play_uri(ctx_uri)
            return {"error": "no match found"}
        except Exception as e:
            return {"error": str(e)}

# ----------------------------
# Bookmarks utilities (unchanged)
# ----------------------------
def get_possible_bookmark_paths():
    env_path = os.environ.get("CHROME_BOOKMARKS_PATH")
    if env_path:
        return [env_path]
    paths = []
    home = Path.home()
    if sys.platform.startswith("win"):
        local = os.environ.get("LOCALAPPDATA") or (home / "AppData" / "Local")
        paths += [
            Path(local) / "Google" / "Chrome" / "User Data" / "Default" / "Bookmarks",
            Path(local) / "Microsoft" / "Edge" / "User Data" / "Default" / "Bookmarks",
            Path(local) / "BraveSoftware" / "Brave-Browser" / "User Data" / "Default" / "Bookmarks",
            Path(local) / "Chromium" / "User Data" / "Default" / "Bookmarks"
        ]
    elif sys.platform.startswith("darwin"):
        paths += [
            home / "Library" / "Application Support" / "Google" / "Chrome" / "Default" / "Bookmarks",
            home / "Library" / "Application Support" / "BraveSoftware" / "Brave-Browser" / "Default" / "Bookmarks",
            home / "Library" / "Application Support" / "Microsoft Edge" / "Default" / "Bookmarks",
            home / "Library" / "Application Support" / "Chromium" / "Default" / "Bookmarks"
        ]
    else:
        paths += [
            home / ".config" / "google-chrome" / "Default" / "Bookmarks",
            home / ".config" / "chromium" / "Default" / "Bookmarks",
            home / ".config" / "brave" / "Default" / "Bookmarks",
            home / ".config" / "microsoft-edge" / "Default" / "Bookmarks"
        ]
    return [str(p) for p in paths if p.exists()]

def load_chrome_bookmarks(bookmarks_path=None):
    candidates = []
    if bookmarks_path:
        candidates = [bookmarks_path]
    else:
        candidates = get_possible_bookmark_paths()
    if not candidates:
        return {"found": False, "error": "No bookmark file found. Set CHROME_BOOKMARKS_PATH if needed.", "paths_checked": get_possible_bookmark_paths()}
    last_err = None
    for path in candidates:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            roots = data.get('roots', {})
            results = []
            def walk(node, parent_path=""):
                ntype = node.get("type")
                if ntype == "url":
                    results.append({"name": node.get("name",""), "url": node.get("url",""), "path": parent_path.strip(" > "), "date_added": node.get("date_added")})
                elif ntype == "folder" or "children" in node:
                    name = node.get("name","")
                    new_parent = (parent_path + " > " + name) if name else parent_path
                    for child in node.get("children",[]):
                        walk(child, new_parent)
            for k in ("bookmark_bar","other","synced"):
                root = roots.get(k)
                if root:
                    walk(root, k)
            if not results:
                for k,v in roots.items():
                    walk(v, k)
            return {"found": True, "path": path, "bookmarks": results}
        except Exception as e:
            last_err = str(e)
    return {"found": False, "error": f"Could not read bookmark file. Last error: {last_err}", "paths_tried": candidates}

# ----------------------------
# NL parser (adds YouTube search and youtube_search action)
# ----------------------------
def parse_nl_to_action(text):
    t = (text or "").strip()
    tl = t.lower().strip()
    # youtube search patterns
    m = re.match(r"(?:open|search)\s+(?:youtube\s+and\s+)?(?:for\s+|for:)?(.+)", tl)
    if m and "youtube" in tl:
        q = m.group(1).strip()
        return {"action":"youtube_search", "params":{"q": q}}
    m = re.match(r"search youtube for (.+)", tl)
    if m:
        q = m.group(1).strip()
        return {"action":"youtube_search", "params":{"q": q}}

    # Spotify
    m = re.match(r"play (.+) on spotify", tl)
    if m:
        return {"action":"spotify_play", "params":{"q": m.group(1).strip()}}
    if re.match(r"^(play|resume) spotify$", tl) or tl == "play music":
        return {"action":"spotify_resume", "params":{}}
    if re.match(r"^(pause|stop) spotify$", tl):
        return {"action":"spotify_pause", "params":{}}
    if re.match(r"^(skip|next|next song|next track)$", tl):
        return {"action":"spotify_next", "params":{}}
    if re.match(r"^(previous|prev|previous song|previous track|back)$", tl):
        return {"action":"spotify_previous", "params":{}}
    m = re.match(r"(rewind|seek back|go back)\s+(\d+)\s*(seconds|secs|s)?", tl)
    if m:
        secs = int(m.group(2))
        return {"action":"spotify_seek", "params":{"offset_seconds": -secs}}

    # Search (google)
    m = re.match(r"(?:search|google search|search google for)\s+(.+)", tl)
    if m:
        query = m.group(1).strip()
        return {"action":"search", "params":{"q": query}}
    m = re.match(r"open (.+) on google", tl)
    if m:
        query = m.group(1).strip()
        return {"action":"search", "params":{"q": query}}

    # direct opens
    if re.match(r"^open youtube$", tl):
        return {"action":"open", "params":{"target":"https://www.youtube.com"}}
    if re.match(r"^open spotify$", tl) or re.match(r"^open spotify web$", tl):
        return {"action":"open", "params":{"target":"spotify"}}
    if "schoology" in tl:
        return {"action":"open", "params":{"target":"https://app.schoology.com"}}

    # bookmarks
    if re.match(r"(list|show|open) bookmarks?$", tl):
        return {"action":"list_bookmarks", "params":{}}
    m = re.match(r"open (?:my )?bookmark(?: titled)? ['\"]?(.+?)['\"]?$", t, flags=re.I)
    if m:
        q = m.group(1).strip()
        return {"action":"open_bookmark", "params":{"q": q}}

    # file explorer
    if re.match(r"(open|show) (file explorer|explorer|file manager|files)$", tl):
        return {"action":"open", "params":{"target":"file_explorer"}}

    # vscode
    m = re.match(r'open\s+["\']?(.+?)["\']?\s+(?:on|in|with)\s+vscode$', t, flags=re.I)
    if m:
        path = m.group(1).strip()
        return {"action":"open_file_vscode", "params":{"path": path}}
    if re.match(r"^(open|launch|start)\s+(vscode|visual studio code)$", tl):
        return {"action":"open_vscode", "params":{}}

    # type, press, click, tts, eye, servo
    m = re.match(r"type (?:the )?(.*)", t, flags=re.I)
    if m:
        return {"action":"type", "params":{"text": m.group(1)}}
    m = re.match(r"press (.+)", t, flags=re.I)
    if m:
        return {"action":"press", "params":{"key": m.group(1).strip()}}
    m = re.match(r"click(?: at)? (\d+)\s*,?\s*(\d+)", t)
    if m:
        return {"action":"click", "params":{"x": int(m.group(1)), "y": int(m.group(2))}}
    m = re.match(r"(say|speak) (.+)", t)
    if m:
        return {"action":"tts", "params":{"text": m.group(2)}}
    m = re.match(r"(set|change) eyes? to (\d{1,3})\s+(\d{1,3})\s+(\d{1,3})", tl)
    if m:
        r,g,b = int(m.group(2)), int(m.group(3)), int(m.group(4))
        r,g,b = max(0,min(255,r)), max(0,min(255,g)), max(0,min(255,b))
        return {"action":"eye", "params":{"r":r,"g":g,"b":b}}
    m = re.match(r"move (\w+) to (\d{1,3})", tl)
    if m:
        name = m.group(1).upper()
        deg = int(m.group(2))
        return {"action":"servo_move", "params":{"name":name,"deg":deg}}

    # run shell but very restricted
    m = re.match(r"run\s+(.+)", tl)
    if m:
        cmd = m.group(1).strip()
        for p in ["echo ", "dir ", "ls ", "ping ", "whoami"]:
            if cmd.startswith(p):
                return {"action":"run_shell","params":{"cmd":cmd.split(), "shell":False}}
        return {"action":"run_shell","params":{"cmd":cmd, "allowed":False}}

    return None

# ----------------------------
# Execution + assistant messages (handles youtube_search)
# ----------------------------
ACTION_WHITELIST = {k: True for k in ACTION_DEFINITIONS.keys()}
SAFE_SHELL_PREFIXES = ["echo ", "dir ", "ls ", "ping ", "whoami"]

def find_and_open_bookmark(query):
    info = load_chrome_bookmarks(None)
    if not info.get("found"):
        return {"found": False, "error": info.get("error", "No bookmarks available")}
    bookmarks = info.get("bookmarks", [])
    q = (query or "").lower().strip()
    if not q:
        return {"found": False, "error": "empty query"}
    exact = [b for b in bookmarks if (b.get("name") or "").lower() == q]
    if exact:
        b = exact[0]
        AGENT.open_app(b["url"])
        return {"found": True, "match_type": "exact_title", "bookmark": b}
    title_sub = [b for b in bookmarks if q in (b.get("name") or "").lower()]
    if title_sub:
        b = title_sub[0]
        AGENT.open_app(b["url"])
        return {"found": True, "match_type": "title_contains", "bookmark": b}
    url_sub = [b for b in bookmarks if q in (b.get("url") or "").lower()]
    if url_sub:
        b = url_sub[0]
        AGENT.open_app(b["url"])
        return {"found": True, "match_type": "url_contains", "bookmark": b}
    parts = [p for p in re.split(r"\s+", q) if p]
    for p in parts:
        any_match = [b for b in bookmarks if p in (b.get("name","") + " " + b.get("url","")).lower()]
        if any_match:
            b = any_match[0]
            AGENT.open_app(b["url"])
            return {"found": True, "match_type": "fuzzy_partial", "bookmark": b}
    return {"found": False, "error": "no match"}

def nl_execute_from_text(text):
    text = (text or "").strip()
    if not text:
        return {"error":"text required","assistant":"I couldn't hear anything."}
    parsed = parse_nl_to_action(text)
    if parsed is None:
        chat_resp = AGENT.chat_single_answer(text)
        return {"action":"chat", "response": chat_resp, "assistant": chat_resp}
    action = parsed['action']
    if not ACTION_WHITELIST.get(action, False):
        chat_resp = AGENT.chat_single_answer(text)
        return {"action":"chat_fallback", "reason":"not_whitelisted", "response": chat_resp, "assistant": chat_resp}
    if action == "run_shell" and not parsed['params'].get("allowed", False):
        chat_resp = AGENT.chat_single_answer(text)
        return {"action":"chat_fallback", "reason":"unsafe_shell", "response": chat_resp, "assistant": chat_resp}

    try:
        # YouTube search
        if action == "youtube_search":
            q = parsed['params'].get("q","")
            if not q:
                return {"action":"youtube_search","result":{"error":"query required"},"assistant":"What should I search for on YouTube?"}
            url = "https://www.youtube.com/results?search_query=" + requests.utils.quote(q)
            res = AGENT.open_app(url)
            assistant = f"Searching YouTube for '{q}'."
            AGENT.record_action("youtube_search")
            return {"action":"youtube_search","result":res,"assistant":assistant}

        # BOOKMARKS
        if action == "list_bookmarks":
            info = load_chrome_bookmarks(None)
            assistant = f"Found {len(info.get('bookmarks',[]))} bookmarks." if info.get("found") else f"No bookmarks: {info.get('error')}"
            AGENT.record_action("list_bookmarks")
            return {"action":"list_bookmarks", "result": info, "assistant": assistant}

        if action == "open_bookmark":
            q = parsed['params'].get("q","")
            res = find_and_open_bookmark(q)
            assistant = f"Opening bookmark matching '{q}'." if res.get("found") else f"Couldn't find a bookmark for '{q}'."
            AGENT.record_action("open_bookmark")
            return {"action":"open_bookmark", "result": res, "assistant": assistant}

        # SPOTIFY
        if action == "spotify_play":
            q = parsed['params'].get("q","")
            if SPOT is None or not SPOT.is_configured():
                AGENT.open_app(f"https://open.spotify.com/search/{requests.utils.quote(q)}")
                assistant = f"Opening Spotify web and searching for '{q}'."
                AGENT.record_action("spotify_play")
                return {"action":"spotify_play", "result":{"fallback":"web_search_opened"}, "assistant": assistant}
            res = SPOT.search_and_play(q)
            assistant = f"Attempting to play '{q}' on Spotify."
            AGENT.record_action("spotify_play")
            return {"action":"spotify_play", "result": res, "assistant": assistant}

        if action == "spotify_pause":
            if SPOT is None or not SPOT.is_configured():
                AGENT.open_app("https://open.spotify.com")
                assistant = "Can't pause — Spotify not configured."
                return {"action":"spotify_pause", "result":{"error":"spotify not configured"}, "assistant": assistant}
            res = SPOT.pause()
            assistant = "Pausing Spotify." if res.get("ok") else f"Could not pause Spotify: {res}"
            AGENT.record_action("spotify_pause")
            return {"action":"spotify_pause", "result": res, "assistant": assistant}

        if action == "spotify_resume":
            if SPOT is None or not SPOT.is_configured():
                AGENT.open_app("https://open.spotify.com")
                assistant = "Opening Spotify web."
                return {"action":"spotify_resume", "result":{"error":"spotify not configured"}, "assistant": assistant}
            res = SPOT.resume()
            assistant = "Resuming Spotify." if res.get("ok") else f"Could not resume Spotify: {res}"
            AGENT.record_action("spotify_resume")
            return {"action":"spotify_resume", "result": res, "assistant": assistant}

        if action == "spotify_next":
            if SPOT is None or not SPOT.is_configured():
                return {"action":"spotify_next","result":{"error":"spotify not configured"},"assistant":"Spotify not configured."}
            res = SPOT.next_track()
            assistant = "Skipping to next track."
            AGENT.record_action("spotify_next")
            return {"action":"spotify_next","result":res,"assistant":assistant}

        if action == "spotify_previous":
            if SPOT is None or not SPOT.is_configured():
                return {"action":"spotify_previous","result":{"error":"spotify not configured"},"assistant":"Spotify not configured."}
            res = SPOT.previous_track()
            assistant = "Going to previous track."
            AGENT.record_action("spotify_previous")
            return {"action":"spotify_previous","result":res,"assistant":assistant}

        if action == "spotify_seek":
            off = parsed['params'].get("offset_seconds", 0)
            if SPOT is None or not SPOT.is_configured():
                return {"action":"spotify_seek","result":{"error":"spotify not configured"},"assistant":"Spotify not configured."}
            play = SPOT.current_playback()
            if play.get("error"):
                return {"action":"spotify_seek","result":play,"assistant":"Couldn't get current playback."}
            progress = play.get("progress_ms") or 0
            newpos = max(0, int(progress) + int(off)*1000)
            res = SPOT.seek(newpos)
            assistant = f"Seeking {off} seconds."
            AGENT.record_action("spotify_seek")
            return {"action":"spotify_seek","result":res,"assistant":assistant}

        # SEARCH
        if action == "search":
            q = parsed['params'].get("q","")
            res = AGENT.web_search(q)
            assistant = f"Searching Google for '{q}'."
            AGENT.record_action("search")
            return {"action":"search","result":res,"assistant":assistant}

        # OPEN generic
        if action == "open":
            tgt = parsed['params'].get("target")
            if tgt == "file_explorer" or (isinstance(tgt,str) and tgt.lower().startswith("explorer")):
                if sys.platform.startswith("win"):
                    res = AGENT.open_app("explorer")
                    assistant = "Opening File Explorer."
                elif sys.platform.startswith("darwin"):
                    res = AGENT.open_app("open .")
                    assistant = "Opening Finder."
                else:
                    res = AGENT.open_app("xdg-open .")
                    assistant = "Opening file manager."
                AGENT.record_action("open")
                return {"action":"open","result":res,"assistant":assistant}

            if isinstance(tgt, str) and tgt.lower() in ("spotify", "spotify web", "open spotify"):
                res = AGENT.open_app("spotify")
                assistant = "Opening Spotify."
                AGENT.record_action("open")
                return {"action":"open","result":res,"assistant":assistant}

            if isinstance(tgt, str) and tgt.startswith("http"):
                res = AGENT.open_app(tgt)
                assistant = f"Opening {tgt}."
                AGENT.record_action("open")
                return {"action":"open","result":res,"assistant":assistant}

            # fallback: pass the target to open_app
            res = AGENT.open_app(str(tgt))
            assistant = f"Opening {tgt}."
            AGENT.record_action("open")
            return {"action":"open","result":res,"assistant":assistant}

        # VSCode open
        if action == "open_vscode":
            cmd = 'code'
            res = AGENT.open_app(cmd)
            assistant = "Opening Visual Studio Code."
            AGENT.record_action("open_vscode")
            return {"action":"open_vscode","result":res,"assistant":assistant}
        if action == "open_file_vscode":
            rawpath = parsed['params'].get("path","")
            p = os.path.expanduser(rawpath)
            if not os.path.isabs(p):
                p = os.path.abspath(p)
            cmd = f'code "{p}"'
            res = AGENT.open_app(cmd)
            assistant = f"Opening {p} in VSCode."
            AGENT.record_action("open_file_vscode")
            return {"action":"open_file_vscode","result":res,"assistant":assistant}

        # TYPE / PRESS / CLICK / TTS / EYE / SERVO
        if action == "type":
            textp = parsed['params'].get("text","")
            res = AGENT.type_text(textp)
            assistant = f"Typing: {textp}"
            AGENT.record_action("type")
            return {"action":"type","result":res,"assistant":assistant}
        if action == "press":
            key = parsed['params'].get("key")
            res = AGENT.press_key(key)
            assistant = f"Pressing {key}."
            AGENT.record_action("press")
            return {"action":"press","result":res,"assistant":assistant}
        if action == "click":
            x = parsed['params'].get("x"); y = parsed['params'].get("y")
            res = AGENT.click(x,y)
            assistant = f"Clicking at {x},{y}."
            AGENT.record_action("click")
            return {"action":"click","result":res,"assistant":assistant}
        if action == "tts":
            txt = parsed['params'].get("text","")
            res = AGENT.tts_say(txt)
            assistant = f"Saying: {txt}"
            AGENT.record_action("tts")
            return {"action":"tts","result":res,"assistant":assistant}
        if action == "eye":
            r = parsed['params'].get("r"); g = parsed['params'].get("g"); b = parsed['params'].get("b")
            res = AGENT.set_eye_color(r,g,b)
            assistant = f"Set eye color to {r},{g},{b}."
            AGENT.record_action("eye")
            return {"action":"eye","result":res,"assistant":assistant}
        if action == "servo_move":
            name = parsed['params'].get("name"); deg = parsed['params'].get("deg")
            res = AGENT.move_servo_deg(name, deg)
            assistant = f"Moving {name} to {deg} degrees."
            AGENT.record_action("servo_move")
            return {"action":"servo_move","result":res,"assistant":assistant}

        # RUN SHELL
        if action == "run_shell":
            cmd = parsed['params'].get("cmd")
            if isinstance(cmd, str) and not any(cmd.startswith(p) for p in SAFE_SHELL_PREFIXES):
                assistant = "That shell command isn't allowed for safety."
                return {"action":"run_shell","result":{"error":"not allowed"},"assistant":assistant}
            res = AGENT.run_shell(cmd, shell=False)
            assistant = f"Ran shell command: {' '.join(cmd) if isinstance(cmd,list) else cmd}"
            AGENT.record_action("run_shell")
            return {"action":"run_shell","result":res,"assistant":assistant}
    except Exception as e:
        return {"error":"execution error","detail": str(e),"assistant": f"Error executing action: {e}"}
    return {"error":"unhandled action","action": action,"assistant":"I couldn't handle that action."}

# ----------------------------
# Flask server for Spotify callback (minimal)
# ----------------------------
app = Flask("saint_agent_ui_desktop")
app.secret_key = os.environ.get("SAINT_SECRET") or os.urandom(24).hex()
SAINT_TOKEN = os.environ.get("SAINT_TOKEN", "MySuperSecretCode")
BIND_HOST = os.environ.get("SAINT_BIND", "127.0.0.1")
BIND_PORT = int(os.environ.get("SAINT_PORT", "8765"))

AGENT = RobotHead()
SPOT = SpotifyController() if HAS_SPOTIPY else None

@app.route("/spotify/callback")
def spotify_callback():
    if SPOT is None or not SPOT.is_configured():
        return f"Spotify not configured. Diagnostics: {SPOT.diagnostics() if SPOT is not None else {'has_spotipy': HAS_SPOTIPY}}", 400
    info = SPOT.authorize_callback(request.args)
    if info.get("error"):
        return f"Auth error: {info.get('error')}", 400
    return "Spotify authorization successful. Close this tab and return to the SAINT UI."

def run_server():
    try:
        print(f"[API] Starting Flask server on {BIND_HOST}:{BIND_PORT}")
        app.run(host=BIND_HOST, port=BIND_PORT, debug=False, use_reloader=False)
    except Exception as e:
        print("[API] Flask server failed:", e)

# ----------------------------
# Voice listener: wake-word -> SR -> execute
# - If agent is speaking and user speaks, stop TTS and listen.
# - Correct frame unpacking for porcupine.
# ----------------------------
class VoiceListener(threading.Thread):
    def __init__(self, gui_ref):
        super().__init__(daemon=True)
        self.gui_ref = gui_ref
        self.running = True

    def run(self):
        print("[VOICE] Voice listener started. Picovoice:", PICOVOICE_AVAILABLE, "SR:", HAS_SR)
        if PICOVOICE_AVAILABLE and PV_PORCUPINE is not None and PAUDIO is not None:
            kw_path = os.environ.get("PICOVOICE_KEYWORD_PATH") or os.path.join("models", "KeyWordModel", "KeyWord.ppn")
            access_key = os.environ.get("PICOVOICE_ACCESS_KEY") or None
            rhino_ctx = os.environ.get("PICOVOICE_CONTEXT_PATH") or os.path.join("models", "RhinoContext", "RhinoContext.rhn")

            porcupine = None
            try:
                # modern PV API requires access_key param
                if access_key:
                    porcupine = PV_PORCUPINE.create(access_key=access_key, keyword_paths=[kw_path])
                else:
                    # attempt create without named arg (older bindings)
                    porcupine = PV_PORCUPINE.create(None, [kw_path])
            except Exception as e:
                print("[VOICE] porcupine create failed:", e)
                porcupine = None

            if porcupine is None:
                print("[VOICE] Porcupine not initialized; will use SR-only fallback.")
            else:
                try:
                    pa = PAUDIO.PyAudio()
                    frame_length = getattr(porcupine, "frame_length", 512)
                    sample_rate = getattr(porcupine, "sample_rate", 16000)
                    stream = pa.open(format=PAUDIO.paInt16,
                                     channels=1,
                                     rate=sample_rate,
                                     input=True,
                                     frames_per_buffer=frame_length)
                    print("[VOICE] Listening for wake-word (Picovoice)...")
                    r_sr = sr.Recognizer() if HAS_SR else None

                    while self.running:
                        # If agent speaking, stop TTS and then listen (user override)
                        if AGENT.is_speaking:
                            # Attempt to stop TTS so we can hear user
                            AGENT.tts_stop()
                            time.sleep(0.05)
                            continue
                        try:
                            pcm_bytes = stream.read(frame_length, exception_on_overflow=False)
                        except Exception:
                            time.sleep(0.01)
                            continue
                        try:
                            pcm = struct.unpack_from("h" * frame_length, pcm_bytes)
                        except Exception:
                            total_samples = len(pcm_bytes) // 2
                            if total_samples >= frame_length:
                                pcm = struct.unpack_from("h" * frame_length, pcm_bytes[:frame_length*2])
                            else:
                                continue
                        try:
                            keyword_index = porcupine.process(pcm)
                        except Exception:
                            try:
                                keyword_index = porcupine.process(pcm_bytes)
                            except Exception as e:
                                continue
                        if keyword_index is not None and keyword_index >= 0:
                            print("[VOICE] Wakeword detected.")
                            captured_text = None
                            if HAS_SR and r_sr is not None:
                                try:
                                    with sr.Microphone() as source:
                                        if AGENT.is_speaking:
                                            AGENT.tts_stop()
                                        r_sr.adjust_for_ambient_noise(source, duration=0.3)
                                        self.gui_ref.set_status("listening (command)...")
                                        audio = r_sr.listen(source, timeout=5, phrase_time_limit=6)
                                        self.gui_ref.set_status("processing...")
                                        try:
                                            captured_text = r_sr.recognize_google(audio)
                                            print("[VOICE] Transcribed:", captured_text)
                                        except sr.UnknownValueError:
                                            print("[VOICE] Could not understand command")
                                            captured_text = None
                                        except Exception as e:
                                            print("[VOICE] SR recognition error:", e)
                                            captured_text = None
                                except Exception as e:
                                    print("[VOICE] SR microphone capture error:", e)
                                    captured_text = None
                            if captured_text is None and PV_RHINO is not None:
                                try:
                                    if os.path.exists(rhino_ctx):
                                        try:
                                            rhino = PV_RHINO.create(context_path=rhino_ctx, access_key=access_key)
                                        except Exception:
                                            try:
                                                rhino = PV_RHINO.create(access_key, rhino_ctx)
                                            except Exception:
                                                rhino = None
                                        if rhino is not None:
                                            inference = None
                                            rh_frame_len = getattr(rhino, "frame_length", 512)
                                            max_iters = int(sample_rate / frame_length * 4)
                                            for _ in range(max_iters):
                                                pcm2 = stream.read(rh_frame_len, exception_on_overflow=False)
                                                try:
                                                    pcm2_unpack = struct.unpack_from("h" * rh_frame_len, pcm2)
                                                except Exception:
                                                    continue
                                                is_final = rhino.process(pcm2_unpack)
                                                if is_final:
                                                    inference = rhino.get_inference()
                                                    break
                                            if inference:
                                                intent_name = inference.get("intent") or inference.get("intent_name", "")
                                                slots = inference.get("slots", {})
                                                captured_text = intent_name + " " + " ".join([f"{k} {v}" for k, v in slots.items()])
                                                print("[VOICE] Rhino inference ->", captured_text)
                                except Exception as e:
                                    print("[VOICE] Rhino attempt failed:", e)
                            if captured_text:
                                AGENT.try_extract_and_save_name(captured_text)
                                res = nl_execute_from_text(captured_text)
                                self.gui_ref.display_action_result(res)
                                speak_msg = res.get("assistant") or res.get("response") or json.dumps(res)
                                if AGENT.output_mode in ("voice","both"):
                                    AGENT.tts_say(str(speak_msg))
                            else:
                                self.gui_ref.display_action_result({"info":"wakeword detected but no command captured", "assistant":"I heard you but didn't catch the command."})
                                if AGENT.output_mode in ("voice","both"):
                                    AGENT.tts_say("I heard you, but I didn't catch the command. Try again.")
                    try:
                        stream.stop_stream(); stream.close(); pa.terminate()
                    except Exception:
                        pass
                except Exception:
                    print("[VOICE] Picovoice audio loop failed:", traceback.format_exc())

        # SpeechRecognition-only fallback loop (no wake-word)
        if HAS_SR:
            r = sr.Recognizer()
            mic = None
            try:
                mic = sr.Microphone()
            except Exception:
                print("[VOICE] Microphone open failed for SR fallback.")
                mic = None
            if mic is None:
                print("[VOICE] No microphone for SR fallback. Voice disabled.")
                return
            while self.running:
                if AGENT.is_speaking:
                    # stop TTS if user interrupts
                    AGENT.tts_stop()
                    time.sleep(0.05)
                    continue
                if AGENT.input_mode != "voice":
                    time.sleep(0.5)
                    continue
                try:
                    with mic as source:
                        self.gui_ref.set_status("listening...")
                        r.adjust_for_ambient_noise(source, duration=0.5)
                        audio = r.listen(source, timeout=6, phrase_time_limit=8)
                    self.gui_ref.set_status("processing...")
                    try:
                        txt = r.recognize_google(audio)
                        print("[VOICE] Heard:", txt)
                        AGENT.try_extract_and_save_name(txt)
                        res = nl_execute_from_text(txt)
                        self.gui_ref.display_action_result(res)
                        speak = res.get("assistant") or res.get("response") or json.dumps(res)
                        if AGENT.output_mode in ("voice","both"):
                            AGENT.tts_say(str(speak))
                    except sr.UnknownValueError:
                        print("[VOICE] Could not understand audio")
                    except Exception as e:
                        print("[VOICE] SR exception:", e)
                    self.gui_ref.set_status("idle")
                except Exception:
                    time.sleep(0.2)
            return
        print("[VOICE] No voice backend available (no Picovoice and no SpeechRecognition). Voice disabled.")

# ----------------------------
# Simple Tkinter GUI (keeps earlier features)
# ----------------------------
class SaintGUI:
    def __init__(self, root):
        self.root = root
        root.title("SAINT Agent (Desktop)")
        root.geometry("920x700")
        self.logged_in = False

        top = ttk.Frame(root, padding=6)
        top.pack(side=tk.TOP, fill=tk.X)
        self.status_label = ttk.Label(top, text="Not logged in")
        self.status_label.pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Login", command=self.login_prompt).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Spotify Login", command=self.spotify_login).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Refresh Bookmarks", command=self.refresh_bookmarks).pack(side=tk.LEFT, padx=4)

        cfg_frame = ttk.LabelFrame(root, text="Configuration", padding=6)
        cfg_frame.pack(fill=tk.X, padx=6, pady=6)
        ttk.Label(cfg_frame, text="Input mode:").grid(row=0,column=0, sticky=tk.W)
        self.input_mode_var = tk.StringVar(value=AGENT.input_mode)
        ttk.Combobox(cfg_frame, textvariable=self.input_mode_var, values=["voice","text"], width=10).grid(row=0,column=1, sticky=tk.W)
        ttk.Label(cfg_frame, text="Output mode:").grid(row=0,column=2, sticky=tk.W, padx=(10,0))
        self.output_mode_var = tk.StringVar(value=AGENT.output_mode)
        ttk.Combobox(cfg_frame, textvariable=self.output_mode_var, values=["voice","text","both"], width=10).grid(row=0,column=3, sticky=tk.W)
        ttk.Button(cfg_frame, text="Save", command=self.save_config).grid(row=0,column=4, padx=6)

        bm_frame = ttk.LabelFrame(root, text="Bookmarks", padding=6)
        bm_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        self.bk_search = ttk.Entry(bm_frame, width=40)
        self.bk_search.pack(anchor=tk.W)
        self.bk_search.bind("<KeyRelease>", lambda e: self.filter_bookmarks())
        self.bk_listbox = tk.Listbox(bm_frame)
        self.bk_listbox.pack(fill=tk.BOTH, expand=True)
        ttk.Button(bm_frame, text="Open Selected", command=self.open_selected_bookmark).pack(anchor=tk.E, pady=4)

        nl_frame = ttk.LabelFrame(root, text="Natural Language → Action / Chat", padding=6)
        nl_frame.pack(fill=tk.BOTH, padx=6, pady=6)
        self.nl_text = tk.Text(nl_frame, height=4)
        self.nl_text.pack(fill=tk.X)
        btn_frame = ttk.Frame(nl_frame)
        btn_frame.pack(fill=tk.X, pady=4)
        ttk.Button(btn_frame, text="Suggest", command=self.suggest_action).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Suggest & Execute", command=self.execute_action).pack(side=tk.LEFT, padx=4)

        log_frame = ttk.LabelFrame(root, text="Result / Log", padding=6)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.log_box = scrolledtext.ScrolledText(log_frame, state=tk.NORMAL)
        self.log_box.pack(fill=tk.BOTH, expand=True)

        self.statbar = ttk.Label(root, text="idle", anchor=tk.W)
        self.statbar.pack(fill=tk.X, side=tk.BOTTOM)

        self.bookmarks_cache = []
        self.vlistener = VoiceListener(self)
        self.vlistener.start()

    def set_status(self, txt):
        self.statbar.config(text=txt)

    def display_action_result(self, res):
        assistant_msg = res.get("assistant") or res.get("response") or res.get("info") or ""
        if assistant_msg:
            self.log_box.insert(tk.END, f"Assistant: {assistant_msg}\n")
        self.log_box.insert(tk.END, json.dumps(res, indent=2) + "\n\n")
        self.log_box.see(tk.END)

    def login_prompt(self):
        token = simpledialog.askstring("Login", "Enter SAINT token:", show="*")
        if token == SAINT_TOKEN:
            self.logged_in = True
            self.status_label.config(text="Logged in")
            messagebox.showinfo("Login", "Authenticated")
        else:
            messagebox.showerror("Login", "Invalid token")

    def spotify_login(self):
        if SPOT is None:
            messagebox.showerror("Spotify", "spotipy not installed or SPOT not configured.")
            return
        url = SPOT.get_auth_url()
        if not url:
            messagebox.showerror("Spotify", f"Could not create auth url. Diagnostics: {SPOT.diagnostics()}")
            return
        webbrowser.open(url)
        messagebox.showinfo("Spotify", "A browser window was opened for Spotify authorization. Complete the flow and return here.")

    def save_config(self):
        AGENT.input_mode = self.input_mode_var.get()
        AGENT.output_mode = self.output_mode_var.get()
        AGENT.save_config()
        messagebox.showinfo("Config", "Config saved")

    def set_eye(self):
        try:
            r = int(self.r_entry.get()); g = int(self.g_entry.get()); b = int(self.b_entry.get())
        except Exception:
            messagebox.showerror("Eye", "Invalid RGB")
            return
        AGENT.set_eye_color(r,g,b)
        self.display_action_result({"action":"eye","result":{"ok":True,"rgb":[r,g,b]}, "assistant": f"Set eye color to {r},{g},{b}."})

    def refresh_bookmarks(self):
        info = load_chrome_bookmarks(None)
        if info.get("found"):
            self.bookmarks_cache = info.get("bookmarks", [])
            self.render_bookmarks(self.bookmarks_cache)
            messagebox.showinfo("Bookmarks", f"Found {len(self.bookmarks_cache)} bookmarks")
        else:
            messagebox.showwarning("Bookmarks", f"No bookmarks: {info.get('error')}")

    def render_bookmarks(self, list_):
        self.bk_listbox.delete(0, tk.END)
        for b in list_:
            display = f"{b.get('name') or b.get('url')} — {b.get('path')}"
            self.bk_listbox.insert(tk.END, display)

    def filter_bookmarks(self):
        q = self.bk_search.get().lower()
        filtered = [b for b in self.bookmarks_cache if q in (b.get("name","") + " " + b.get("url","") + " " + b.get("path","")).lower()]
        self.render_bookmarks(filtered)

    def open_selected_bookmark(self):
        sel = self.bk_listbox.curselection()
        if not sel:
            messagebox.showwarning("Open", "Select a bookmark first")
            return
        idx = sel[0]
        q = self.bk_search.get().lower()
        filtered = [b for b in self.bookmarks_cache if q in (b.get("name","") + " " + b.get("url","") + " " + b.get("path","")).lower()]
        b = filtered[idx]
        res = AGENT.open_app(b.get("url"))
        assistant = f"Opening {b.get('name') or b.get('url')}"
        self.display_action_result({"opened": b.get('url'), "result": res, "assistant": assistant})

    def suggest_action(self):
        txt = self.nl_text.get("1.0", tk.END).strip()
        if not txt:
            messagebox.showwarning("Suggest", "Enter text first")
            return
        parsed = parse_nl_to_action(txt)
        if parsed is None:
            self.display_action_result({"understood": False, "message":"Could not parse. I can chat instead."})
            return
        action = parsed.get("action")
        allowed = ACTION_WHITELIST.get(action, False)
        if action == "run_shell":
            allowed = allowed and any(parsed['params'].get("cmd","").startswith(p) for p in SAFE_SHELL_PREFIXES)
        self.display_action_result({"understood": True, "suggestion": parsed, "allowed": allowed})

    def execute_action(self):
        txt = self.nl_text.get("1.0", tk.END).strip()
        if not txt:
            messagebox.showwarning("Execute", "Enter text first")
            return
        res = nl_execute_from_text(txt)
        self.display_action_result(res)
        speak = res.get("assistant") or res.get("response") or json.dumps(res)
        if AGENT.output_mode in ("voice","both"):
            AGENT.tts_say(str(speak))

# ----------------------------
# Start everything
# ----------------------------
def main():
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    root = tk.Tk()
    app = SaintGUI(root)
    try:
        app.refresh_bookmarks()
    except Exception:
        pass
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Exiting")
    finally:
        try:
            app.vlistener.running = False
        except Exception:
            pass

if __name__ == "__main__":
    main()

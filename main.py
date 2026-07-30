import os, re, json, asyncio, tempfile, subprocess
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import google.generativeai as genai
import edge_tts

app = FastAPI()
UPLOAD_DIR = Path("uploads"); UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path("outputs"); OUTPUT_DIR.mkdir(exist_ok=True)

app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/static", StaticFiles(directory="static"), name="static")

VOICES = {
    "vi-VN-HoaiMyNeural": "Tiếng Việt - Nữ (Hoài My · Neural)",
    "vi-VN-NamMinhNeural": "Tiếng Việt - Nam (Nam Minh · Neural)",
}

def parse_srt(content: str):
    blocks = re.split(r'\n\s*\n', content.strip())
    subs = []
    for b in blocks:
        lines = b.strip().splitlines()
        if len(lines) < 3: continue
        try:
            idx = int(lines[0].strip())
        except:
            continue
        timing = lines[1]
        text = " ".join(lines[2:]).strip()
        m = re.match(r'(\d+:\d+:\d+[,\.]\d+)\s*-->\s*(\d+:\d+:\d+[,\.]\d+)', timing)
        if not m: continue
        subs.append({"index": idx, "start": m.group(1), "end": m.group(2), "text": text})
    return subs

def ts_to_sec(ts: str) -> float:
    ts = ts.replace(',', '.')
    h, m, rest = ts.split(':')
    return int(h)*3600 + int(m)*60 + float(rest)

def sec_to_ts(sec: float) -> str:
    h = int(sec // 3600); m = int((sec % 3600) // 60); s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace('.', ',')

async def extract_audio(video_path: str, audio_path: str):
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", audio_path]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
    await proc.wait()

async def transcribe_whisper(audio_path: str) -> str:
    import whisper
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, language="zh", task="transcribe")
    segs = result["segments"]
    lines = []
    for i, s in enumerate(segs, 1):
        lines.append(str(i))
        lines.append(f"{sec_to_ts(s['start'])} --> {sec_to_ts(s['end'])}")
        lines.append(s["text"].strip())
        lines.append("")
    return "\n".join(lines)

async def translate_with_gemini(api_key: str, subs: list) -> list:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    texts = [s["text"] for s in subs]
    batch = "\n---\n".join(texts)
    prompt = f"""Dịch các đoạn tiếng Trung sau sang tiếng Việt tự nhiên, giữ nguyên ý nghĩa, phong cách tự nhiên.
Mỗi đoạn cách nhau bằng dấu ---. Trả về ĐÚNG số đoạn, mỗi đoạn cách nhau bằng ---. Không thêm giải thích.

{batch}"""
    resp = model.generate_content(prompt)
    translated = [t.strip() for t in resp.text.split("---")]
    result = []
    for i, s in enumerate(subs):
        result.append({**s, "translated": translated[i] if i < len(translated) else s["text"]})
    return result

def build_srt(subs: list, use_translated=True) -> str:
    lines = []
    for s in subs:
        lines.append(str(s["index"]))
        lines.append(f"{s['start']} --> {s['end']}")
        lines.append(s["translated"] if use_translated else s["text"])
        lines.append("")
    return "\n".join(lines)

async def generate_tts_audio(subs: list, voice: str, output_dir: Path) -> list:
    audio_files = []
    for s in subs:
        out = output_dir / f"tts_{s['index']:04d}.mp3"
        communicate = edge_tts.Communicate(s["translated"], voice)
        await communicate.save(str(out))
        audio_files.append({"path": str(out), "start": ts_to_sec(s["start"]), "end": ts_to_sec(s["end"]), "index": s["index"]})
    return audio_files

async def merge_audio_video(video_path: str, audio_files: list, srt_path: str, output_path: str, job_dir: Path):
    duration_cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path]
    proc = await asyncio.create_subprocess_exec(*duration_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    out, _ = await proc.communicate()
    total_dur = float(json.loads(out)["format"]["duration"])

    silent_base = str(job_dir / "silent_base.wav")
    cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo", "-t", str(total_dur), silent_base]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
    await proc.wait()

    filter_parts = [f"[0:a]"]
    inputs = ["-i", silent_base]
    delays = []
    for idx, af in enumerate(audio_files):
        inputs += ["-i", af["path"]]
        delay_ms = int(af["start"] * 1000)
        delays.append(f"[{idx+1}:a]adelay={delay_ms}|{delay_ms}[d{idx}]")

    filter_complex = "; ".join(delays)
    amix_inputs = "".join([f"[d{i}]" for i in range(len(audio_files))])
    filter_complex += f"; [0:a]{amix_inputs}amix=inputs={len(audio_files)+1}:duration=first:dropout_transition=0[aout]"

    mixed_audio = str(job_dir / "mixed.wav")
    cmd = ["ffmpeg", "-y"] + inputs + ["-filter_complex", filter_complex, "-map", "[aout]", mixed_audio]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
    await proc.wait()

    cmd = [
        "ffmpeg", "-y", "-i", video_path, "-i", mixed_audio,
        "-vf", f"subtitles={srt_path}:force_style='FontName=Arial,FontSize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2'",
        "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        "-c:a", "aac", "-b:a", "128k",
        output_path
    ]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
    await proc.wait()

@app.get("/")
async def index():
    return FileResponse("static/index.html")

@app.get("/voices")
async def get_voices():
    return JSONResponse(VOICES)

@app.post("/translate")
async def translate_video(
    video: UploadFile = File(...),
    api_key: str = Form(...),
    voice: str = Form("vi-VN-HoaiMyNeural"),
):
    job_id = tempfile.mktemp(dir="", prefix="job_").lstrip("/")
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(exist_ok=True)

    video_path = str(job_dir / video.filename)
    with open(video_path, "wb") as f:
        f.write(await video.read())

    audio_path = str(job_dir / "audio.wav")
    await extract_audio(video_path, audio_path)

    srt_raw = await transcribe_whisper(audio_path)
    subs = parse_srt(srt_raw)
    if not subs:
        raise HTTPException(400, "Không tách được phụ đề từ video")

    subs = await translate_with_gemini(api_key, subs)

    srt_path = str(job_dir / "translated.srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(build_srt(subs))

    tts_dir = job_dir / "tts"; tts_dir.mkdir(exist_ok=True)
    audio_files = await generate_tts_audio(subs, voice, tts_dir)

    output_filename = f"dubbed_{job_id}.mp4"
    output_path = str(OUTPUT_DIR / output_filename)
    await merge_audio_video(video_path, audio_files, srt_path, output_path, job_dir)

    return JSONResponse({"status": "done", "file": f"/outputs/{output_filename}", "job_id": job_id})

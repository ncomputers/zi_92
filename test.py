#test.py
import subprocess
import threading

rtsp_urls = [
    "rtsp://rapidmistryudr%40gmail.com:Rapid%4018@192.168.1.231:554/stream1",
    "rtsp://rapidmistryudr%40gmail.com:Rapid%4018@192.168.1.231:554/stream1"  # you can update to a second stream
]

ffmpeg_cmd_template = lambda url: [
    "ffmpeg",
    "-hwaccel", "cuda",
    "-c:v", "h264_cuvid",
    "-rtsp_transport", "tcp",
    "-i", url,
    "-t", "10",  # run for 10 seconds
    "-f", "null", "-"
]

def run_ffmpeg_stream(url):
    print(f"🚀 Starting FFmpeg for: {url}")
    process = subprocess.Popen(ffmpeg_cmd_template(url), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line.strip())
    process.wait()

# Start both streams in parallel
threads = []
for url in rtsp_urls:
    t = threading.Thread(target=run_ffmpeg_stream, args=(url,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()

print("✅ Both streams completed.")

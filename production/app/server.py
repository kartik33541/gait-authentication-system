from http.server import BaseHTTPRequestHandler, HTTPServer
from infer_identity import predict_person
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "..", "received_gait.csv")


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)

            csv_text = post_data.decode("utf-8")

            # 1️⃣ Save received CSV
            with open(CSV_PATH, "w", encoding="utf-8") as f:
                f.write(csv_text)

            # 2️⃣ Run inference
            person = predict_person(CSV_PATH)

            print("✅ IDENTIFIED:", person)

            # 3️⃣ Respond to phone
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(person.encode("utf-8"))

        except Exception as e:
            print("❌ ERROR:", e)
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"ERROR")


server = HTTPServer(("0.0.0.0", 8000), Handler)
print("🚀 Server running on port 8000...")
server.serve_forever()



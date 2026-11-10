"""Optional legacy loopback HTTP adapter; Isaac Lab uses deployment.Policy directly."""
import argparse
import base64
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import numpy as np
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
from deployment_script.deployment import Policy as LocalPolicy

class Policy(LocalPolicy):
    """Compatibility adapter for the optional legacy HTTP service."""
    def predict(self, payload):
        images = []
        for item in payload["images"]:
            shape = item["shape"]
            if len(shape) != 3 or shape[-1] != 3 or not all(isinstance(n, int) and 0 < n <= 4096 for n in shape):
                raise ValueError("Invalid RGB shape")
            data = base64.b64decode(item["data"], validate=True)
            if len(data) != int(np.prod(shape)):
                raise ValueError("RGB byte count mismatch")
            images.append(np.frombuffer(data, dtype=np.uint8).reshape(shape))
        return {"actions": super().predict(images, payload["state"]).tolist()}

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text(encoding="utf-8"));policy=Policy(cfg)
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                if self.path!="/predict": raise ValueError("Use /predict")
                size=int(self.headers.get("Content-Length",0))
                if not 0<size<32*1024*1024: raise ValueError("Invalid request size")
                result=policy.predict(json.loads(self.rfile.read(size)));status=200
            except Exception as exc:
                result={"error":str(exc)};status=400
            data=json.dumps(result,allow_nan=False).encode();self.send_response(status)
            self.send_header("Content-Type","application/json");self.send_header("Content-Length",str(len(data)))
            self.end_headers();self.wfile.write(data)
    server=HTTPServer(("127.0.0.1",cfg.get("port",8765)),Handler)
    print("Student ready at http://127.0.0.1:%s/predict" % cfg.get("port",8765),flush=True)
    try: server.serve_forever()
    finally: server.server_close()

if __name__=="__main__": main()

# Haran AI

Agro-pastoral guardian for Tunisian family farms. Photograph an olive leaf to see whether the tree is sick, how severe the damage is, what to do, and when to check again. Watch the barn camera and get an alert if a cow may be sick.

## What the jury can try

1. **Grove watch** — choose an olive-leaf photo. Haran measures the damaged area and returns Low, Medium, or High severity, plus a solution and a check-again date.
2. **Barn watch** — open the live camera. Haran says whether a cow looks well, is resting, or may be sick, and notifies the farmer when a cow may be sick.
3. **Farm book** — each leaf check is saved with the place, the solution, and the next visit date.

## Run it

The API key is not in this repository. Copy the example file and add your own Groq key:

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
python main.py
```

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open http://localhost:5173.

The barn camera uses YOLOv8n. The first time the camera starts, the small `yolov8n.pt` weights are downloaded into `backend/`.

# Farm-Vision(Crop Doc)

Mobile-first crop disease detection with a **Next.js frontend** and a **FastAPI backend** powered by a **fastai model** plus a disease knowledge base (`disease_info.json`). Capture or upload a leaf image and get **disease name, cause, prevention, treatment, and confidence**.

Supports **Google sign-in** or **guest mode**. History is saved for signed-in users in **Firestore**.

---

## 📂 Monorepo layout

```
plant-health-ai/
├─ api/                 # FastAPI server (model + JSON metadata)
│  ├─ main.py
│  ├─ requirements.txt
│  ├─ disease_info.json
│  └─ models/
│     └─ model.pkl     # exported fastai model (not committed)
├─ web/                 # Next.js 14 App Router UI
│  ├─ app/
│  │  ├─ layout.js
│  │  ├─ page.js
│  │  └─ globals.css
│  ├─ public/
│  │  └─ favicon.jpeg  # website logo/favicon
│  ├─ package.json
│  ├─ next.config.js
│  ├─ postcss.config.js
│  ├─ tailwind.config.js
│  └─ .env.local.example
└─ research/
   └─ Plant_disease.ipynb
```

---

## 🔧 Prerequisites

* **Node.js 18+** and npm for the frontend.
* **Python 3.10+** for the backend.
* A **Firebase project** (Authentication + Firestore).
* A **fastai exported model (`.pkl`)** placed under `api/models/` and referenced by `MODEL_PATH` in `main.py`.

---

## 🚀 1) Backend: FastAPI server

From the `api/` folder:

### Create venv and install dependencies

**macOS/Linux**

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

**Windows**

```bash
py -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt
```

### Model and metadata

* Put your model file under `api/models/` and ensure `MODEL_PATH` in `main.py` points to it.
* Keep `disease_info.json` in `api/` (already present).

### Run the server

```bash
uvicorn main:app --reload --port 8000
```

API available at: **[http://localhost:8000](http://localhost:8000)**

* **Health endpoint:** `/` (if defined)
* **Prediction endpoint:** `POST /predict` with file field `"file"`

### Test the endpoint

```bash
curl -X POST -F "file=@sample.jpg" http://localhost:8000/predict
```

**Notes**

* CORS is enabled for dev in `main.py`. Tighten `allow_origins` in production.
* If the model cannot be found or loaded, the app will raise on startup.

---

## 💻 2) Frontend: Next.js app

From the `web/` folder:

### Install dependencies

```bash
npm install
```

### Configure environment

```bash
cp .env.local.example .env.local
```

Fill in:

* `NEXT_PUBLIC_FIREBASE_*` with Firebase web config.
* `NEXT_PUBLIC_API_ENDPOINT=http://localhost:8000/predict` (or your deployed API URL).

### Development server

```bash
npm run dev
```

App available at: **[http://localhost:3000](http://localhost:3000)**

### Build and start

```bash
npm run build
npm start
```

**Key behaviors**

* Google Sign-In or “Continue as Guest”. Guest mode disables history save.
* History stored at `users/{uid}/scans` in Firestore.
* Mobile-first UI supports camera capture + gallery upload.

---

## 🌍 3) Launch (Local + Deployment)

### Local end-to-end

```bash
# API
cd api && uvicorn main:app --reload --port 8000

# Web
cd web && npm run dev
```

Open **[http://localhost:3000](http://localhost:3000)** (ensure API host is reachable).

### Deployment

**Backend:**

* Render, Fly.io, Railway, Azure App Service, or VM with uvicorn/gunicorn + Nginx.
* Persist `model.pkl` in container/volume and align `MODEL_PATH`.

**Frontend:**

* Deploy `web/` to Vercel or Netlify.
* Set `NEXT_PUBLIC_API_ENDPOINT` to deployed API’s HTTPS URL.
* Add your web host to Firebase authorized domains.

**Security tips**

* Never commit secrets. Use environment variables.
* Restrict Firestore rules to `users/{uid}/scans`.

---

## 📑 4) API Contract

### `POST /predict`

**Body:** `multipart/form-data` with field `file` (`image/jpeg` or `image/png`)

**Response:**

```json
{
  "predicted_class": "Tomato___Late_blight",
  "confidence": 0.95,
  "details": {
    "name_of_species": "Tomato",
    "diseased_or_healthy": "Diseased",
    "disease_name": "Late Blight",
    "cause": "Phytophthora infestans",
    "prevention": "Use resistant varieties, crop rotation",
    "treatment": "Apply fungicide"
  }
}
```

Frontend normalizes this into:
`status`, `diseaseName`, `cause`, `prevention`, `treatment`, and `name_of_species`.

---

## 🛠 5) Troubleshooting

* **Frontend cannot reach API**

  * Check `NEXT_PUBLIC_API_ENDPOINT`.
  * Verify CORS in `api/main.py`.

* **All predictions show Diseased**

  * Ensure backend returns `"Healthy"`/`"None"` where applicable.

* **History not saving**

  * Confirm user is signed-in.
  * Verify Firestore rules + `users/{uid}/scans`.

* **Mobile camera not opening**

  * Use `capture="environment"` input.
  * On iOS Safari: requires HTTPS + permissions.

---

## 📜 6) Scripts quick ref

**API**

```bash
cd api && source .venv/bin/activate && uvicorn main:app --reload --port 8000
```

**Web**

```bash
cd web && npm install && npm run dev
```

---


## 🙏 Acknowledgements

* **fastai** → model training/inference
* **FastAPI** → backend serving
* **Next.js + Tailwind** → frontend (mobile-first)
* **Firebase** → authentication + Firestore storage

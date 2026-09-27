import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import { Circle, MapContainer, Marker, TileLayer, useMap, useMapEvents } from "react-leaflet";
import L from "leaflet";
import { Camera, LoaderCircle, Volume2 } from "lucide-react";
import { englishSpeechText, speakDerja, unlockDerja } from "../utils/derjaAudio.js";
import { saveFarmCheck } from "../utils/farmBook.js";
import "leaflet/dist/leaflet.css";

const API_BASE = import.meta.env.VITE_API_URL || "";
const CONNECTION_FAILED = "The connection failed. Please photograph the olive leaf again.";
const HEALTHY_SPEECH =
  "Farmer, this tree looks healthy. Keep watering and light pruning. No spray is needed today.";
const SICK_SPEECH = "Farmer, this tree looks sick. Ask an expert before you spray.";
const UNSURE_SPEECH =
  "Farmer, I cannot tell from this photo. Hold one leaf close to the camera so the leaf fills the picture.";

function englishDiseaseLabel(name, status) {
  const raw = typeof name === "string" ? name : "";
  if (status === "healthy") {
    return "Healthy tree";
  }
  if (status === "uncertain") {
    return "Take a closer photo";
  }
  if (/spilocaea|peacock/i.test(raw)) {
    return "Peacock spot (Spilocaea oleagina)";
  }
  if (/pseudomonas|olive knot|\bknot\b/i.test(raw)) {
    return "Olive knot (Pseudomonas savastanoi)";
  }
  if (/verticillium|wilt/i.test(raw)) {
    return "Olive wilt (Verticillium dahliae)";
  }
  if (/aculus|mite/i.test(raw)) {
    return "Olive mite damage";
  }
  if (/anthracnose|colletotrichum/i.test(raw)) {
    return "Anthracnose";
  }
  if (raw && !/[\u0600-\u06FF]/.test(raw)) {
    return raw;
  }
  return "Olive leaf disease";
}

const MORNAG_GROVE = { lat: 36.6742, lng: 10.2854 };
const ORCHARD_ZOOM = 19;
const ESRI_SATELLITE =
  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";

function makeTreeIcon(status) {
  if (status === "target") {
    return L.divIcon({
      className: "tree-marker",
      html: `<div style="width:36px;height:36px;border-radius:50%;background:#f8fafc;border:5px solid #10b981;box-shadow:0 0 0 4px rgba(15,23,42,0.55)"></div>`,
      iconSize: [36, 36],
      iconAnchor: [18, 18],
    });
  }
  const fill = status === "healthy" ? "#10b981" : status === "uncertain" ? "#e2e8f0" : "#ef4444";
  return L.divIcon({
    className: "tree-marker",
    html: `<div style="width:28px;height:28px;border-radius:50%;background:${fill};border:4px solid #f8fafc;box-shadow:0 0 0 3px ${fill}"></div>`,
    iconSize: [28, 28],
    iconAnchor: [14, 14],
  });
}

function Recenter({ lat, lng }) {
  const map = useMap();
  useEffect(() => {
    map.setView([lat, lng], ORCHARD_ZOOM);
  }, [lat, lng, map]);
  return null;
}

function GrovePlacement({ pin, onMove }) {
  useMapEvents({
    click(event) {
      onMove({ lat: event.latlng.lat, lng: event.latlng.lng });
    },
  });

  return (
    <Marker
      position={[pin.lat, pin.lng]}
      draggable
      zIndexOffset={1000}
      icon={makeTreeIcon("target")}
      eventHandlers={{
        dragend(event) {
          const point = event.target.getLatLng();
          onMove({ lat: point.lat, lng: point.lng });
        },
      }}
    />
  );
}

function readCoords() {
  return new Promise((resolve) => {
    if (!("geolocation" in navigator)) {
      resolve(MORNAG_GROVE);
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (position) => {
        resolve({
          lat: position.coords.latitude,
          lng: position.coords.longitude,
        });
      },
      () => resolve(MORNAG_GROVE),
      { enableHighAccuracy: true, timeout: 4000, maximumAge: 60000 },
    );
  });
}

export default function FieldGuardian() {
  const inputRef = useRef(null);
  const placedByFarmer = useRef(false);
  const [pin, setPin] = useState(MORNAG_GROVE);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [preview, setPreview] = useState("");
  const [latest, setLatest] = useState(null);
  const [trees, setTrees] = useState([]);

  useEffect(() => {
    let cancelled = false;
    readCoords().then((value) => {
      if (!cancelled && !placedByFarmer.current) {
        setPin(value);
      }
    });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (latest?.voice_script) {
      speakDerja(latest.voice_script);
    }
  }, [latest]);

  const mapCenter = useMemo(() => [pin.lat, pin.lng], [pin]);

  function placePin(next) {
    placedByFarmer.current = true;
    setPin(next);
  }

  async function handleFile(event) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) {
      return;
    }

    unlockDerja();
    setError("");
    setLoading(true);
    setPreview(URL.createObjectURL(file));

    const here = pin;
    const form = new FormData();
    form.append("file", file);
    form.append("lat", String(here.lat));
    form.append("lng", String(here.lng));

    try {
      const response = await fetch(`${API_BASE}/api/diagnose`, {
        method: "POST",
        body: form,
      });
      if (!response.ok) {
        throw new Error("diagnose-failed");
      }
      const data = await response.json();
      const diagnosisStatus = data.diagnosis?.status || "uncertain";
      const speechFallback =
        diagnosisStatus === "healthy"
          ? HEALTHY_SPEECH
          : diagnosisStatus === "uncertain"
            ? UNSURE_SPEECH
            : SICK_SPEECH;
      const spoken = englishSpeechText(
        data.voice_script || data.derja_spoken_prescription || "",
        speechFallback,
      );
      const tree = {
        id: `${Date.now()}-${trees.length}`,
        status: diagnosisStatus,
        disease: englishDiseaseLabel(data.diagnosis?.disease_name_ar, data.diagnosis?.status),
        severity: data.diagnosis?.severity || "Low",
        expert: Boolean(data.diagnosis?.requires_expert_confirmation),
        voice: spoken,
        lat: here.lat,
        lng: here.lng,
        radius: diagnosisStatus === "sick" ? data.map?.contagion_radius_meters ?? 12 : 0,
      };
      setLatest({ ...data, voice_script: spoken });
      setTrees((current) => [...current, tree]);
      saveFarmCheck({
        lat: here.lat,
        lng: here.lng,
        status: diagnosisStatus,
        disease: tree.disease,
        severity: data.diagnosis?.severity || "Low",
        solution: data.solution || spoken,
        checkAgainDays: data.check_again_days,
        damagePercent: data.leaf_damage_percent,
      });
    } catch {
      setError(CONNECTION_FAILED);
      speakDerja(CONNECTION_FAILED);
    } finally {
      setLoading(false);
    }
  }

  const status = latest?.diagnosis?.status;
  const cardTone = status === "healthy" ? "healthy" : status === "sick" ? "sick" : "";

  return (
    <section className="guardian-panel field-panel">
      <label className="giant-action" onPointerDown={unlockDerja}>
        <Camera size={42} strokeWidth={2.4} aria-hidden="true" />
        <span>Photograph an olive leaf</span>
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          onChange={handleFile}
        />
      </label>

      {loading && (
        <div className="status-card loading-card" role="status">
          <LoaderCircle className="spin" size={40} />
          <p>Haran is checking the leaf...</p>
        </div>
      )}

      {error && (
        <div className="status-card sick-card">
          <p>{error}</p>
        </div>
      )}

      {latest && (
        <article className={`status-card ${cardTone}-card`}>
          {preview && <img className="leaf-preview" src={preview} alt="Photographed olive leaf" />}
          <p className="status-kicker">
            {status === "healthy"
              ? "Healthy"
              : status === "uncertain"
                ? "Not sure"
                : latest.diagnosis.severity === "Low"
                  ? "Low severity"
                  : "Sick tree"}
          </p>
          <h2>{englishDiseaseLabel(latest.diagnosis.disease_name_ar, status)}</h2>
          {status === "sick" && <p className="severity-line">Severity: {latest.diagnosis.severity}</p>}
          {latest.leaf_damage_percent != null && (
            <p className="severity-line">Leaf area affected: {latest.leaf_damage_percent}%</p>
          )}
          <p className="prescription">{latest.solution || latest.voice_script}</p>
          {latest.diagnosis.requires_expert_confirmation && (
            <p className="expert-note">Ask a farm expert to confirm before you spray.</p>
          )}
          <button
            type="button"
            className="replay-btn"
            onClick={() => speakDerja(latest.voice_script)}
          >
            <Volume2 size={32} />
            Hear the advice again
          </button>
        </article>
      )}

      <div className="map-wrap">
        <MapContainer
          center={mapCenter}
          zoom={ORCHARD_ZOOM}
          scrollWheelZoom={false}
          className="orchard-map"
        >
          <TileLayer url={ESRI_SATELLITE} attribution="Tiles &copy; Esri" maxZoom={20} />
          <Recenter lat={mapCenter[0]} lng={mapCenter[1]} />
          <GrovePlacement pin={pin} onMove={placePin} />
          {trees.map((tree) => (
            <Fragment key={tree.id}>
              <Marker position={[tree.lat, tree.lng]} icon={makeTreeIcon(tree.status)} />
              {tree.status === "sick" && tree.radius > 0 && (
                <Circle
                  center={[tree.lat, tree.lng]}
                  radius={tree.radius}
                  pathOptions={{
                    color: "#ef4444",
                    fillColor: "#ef4444",
                    fillOpacity: 0.28,
                    weight: 3,
                  }}
                />
              )}
            </Fragment>
          ))}
        </MapContainer>
      </div>
    </section>
  );
}

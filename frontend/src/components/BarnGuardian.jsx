import { useEffect, useRef, useState } from "react";
import { Volume2 } from "lucide-react";
import { speakDerja } from "../utils/derjaAudio.js";

const API_BASE = import.meta.env.VITE_API_URL || "";

const WELL_SCRIPT = "This cow does not look sick.";
const RESTING_SCRIPT = "The cow is resting. Lying down is normal. It is not sickness.";
const UNWELL_SCRIPT = "Check this cow. She may be sick. Go and look.";
const ALERT_TITLE = "Haran: a cow may be sick";

const DARK_SCRIPT =
  "The camera picture is black. Slide open the camera cover and point it at the cow.";

function notifyFarmer(body) {
  const text = body || UNWELL_SCRIPT;
  if ("vibrate" in navigator) {
    navigator.vibrate([400, 150, 400, 150, 400]);
  }
  if (!("Notification" in window) || Notification.permission !== "granted") {
    return;
  }
  const note = new Notification(ALERT_TITLE, {
    body: text,
    requireInteraction: true,
    tag: "haran-cow-sick",
  });
  note.onclick = () => {
    window.focus();
    note.close();
  };
}

export default function BarnGuardian() {
  const [status, setStatus] = useState({
    camera: "starting",
    cows: 0,
    health: "none",
    reason: "",
    distressed: false,
    dark: false,
  });
  const [streamUrl] = useState(() => `${API_BASE}/api/cow-stream?watch=${Date.now()}`);
  const spokenState = useRef("");
  const alerted = useRef(false);
  const [alertsOn, setAlertsOn] = useState(
    typeof Notification !== "undefined" && Notification.permission === "granted",
  );

  useEffect(() => {
    const previousTitle = document.title;
    return () => {
      document.title = previousTitle;
    };
  }, []);

  async function enableAlerts() {
    if (!("Notification" in window)) {
      return;
    }
    const permission = await Notification.requestPermission();
    setAlertsOn(permission === "granted");
  }

  useEffect(() => {
    let stop = false;

    const tick = async () => {
      try {
        const response = await fetch(`${API_BASE}/api/cow-status`);
        if (!response.ok) {
          return;
        }
        const data = await response.json();
        if (!stop) {
          setStatus(data);
        }
      } catch {
        if (!stop) {
          setStatus((current) => ({ ...current, camera: "missing" }));
        }
      }
    };

    tick();
    const timer = setInterval(tick, 400);
    return () => {
      stop = true;
      clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    if (status.health === "unwell" && !alerted.current) {
      alerted.current = true;
      document.title = "COW MAY BE SICK — Haran";
      notifyFarmer(status.reason || UNWELL_SCRIPT);
      speakDerja(UNWELL_SCRIPT);
      spokenState.current = UNWELL_SCRIPT;
    }
    if (status.health !== "unwell") {
      alerted.current = false;
      if (document.title.startsWith("COW MAY BE SICK")) {
        document.title = "Haran AI";
      }
    }
  }, [status.health, status.reason]);

  useEffect(() => {
    let message = "";
    if (status.health === "unwell") {
      return;
    } else if (status.dark) {
      message = DARK_SCRIPT;
    } else if (status.health === "well") {
      message = status.reason || WELL_SCRIPT;
    } else if (status.health === "resting") {
      message = RESTING_SCRIPT;
    }
    if (message && spokenState.current !== message) {
      spokenState.current = message;
      speakDerja(message);
    }
  }, [status]);

  const unwell = status.health === "unwell";
  const dark = status.dark && !unwell;
  const missing = status.camera === "missing";
  const cardTone = unwell ? "sick" : "healthy";
  const kicker = unwell ? "Barn alert" : "Barn watch";
  let title = "The camera is watching the cows.";
  let detail = `Cows in view: ${status.cows}.`;
  let spoken = "The camera is watching the cows.";

  if (unwell) {
    title = "This cow may be sick.";
    detail = status.reason || UNWELL_SCRIPT;
    spoken = detail;
  } else if (status.health === "well") {
    title = "This cow does not look sick.";
    detail = status.reason || WELL_SCRIPT;
    spoken = detail;
  } else if (status.health === "resting") {
    title = "The cow is resting.";
    detail = RESTING_SCRIPT;
    spoken = RESTING_SCRIPT;
  } else if (status.cows > 0) {
    title = "Checking if this cow is sick.";
    detail = "Haran looks for wounds, swelling, and a cow that has stopped moving. Lying down to rest is not sickness.";
    spoken = "Checking if this cow is sick.";
  } else if (missing) {
    title = "No camera was found.";
    detail = "Plug in a camera, then open Barn watch again.";
    spoken = detail;
  } else if (dark) {
    title = "The camera is on, but the picture is black.";
    detail = DARK_SCRIPT;
    spoken = DARK_SCRIPT;
  }

  return (
    <section className="guardian-panel barn-panel">
      <div className={`status-card ${cardTone}-card barn-alert`}>
        <p className="status-kicker">{kicker}</p>
        <h2>{title}</h2>
        <p className="prescription">{detail}</p>
        {!alertsOn && (
          <button type="button" className="replay-btn danger" onClick={enableAlerts}>
            Turn on sick-cow alerts
          </button>
        )}
        <button type="button" className="replay-btn" onClick={() => speakDerja(spoken)}>
          <Volume2 size={32} />
          Hear the advice again
        </button>
      </div>

      <figure className="stream-frame">
        <img src={streamUrl} alt="Live cow camera" className="cow-stream" />
        <figcaption>Live barn camera. Haran says if the cow looks sick or well.</figcaption>
      </figure>
    </section>
  );
}

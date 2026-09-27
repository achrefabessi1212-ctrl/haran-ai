const ARABIC_SCRIPT = /[\u0600-\u06FF]/;
const ENGLISH_SPEECH_FALLBACK =
  "Farmer, Haran has an update. Please check the screen and ask an expert before you spray.";

let speechUnlocked = false;

export function englishSpeechText(text, fallback = ENGLISH_SPEECH_FALLBACK) {
  if (typeof text !== "string") {
    return fallback;
  }
  const cleaned = text.replace(/\s+/g, " ").trim();
  if (!cleaned || ARABIC_SCRIPT.test(cleaned)) {
    return fallback;
  }
  return cleaned;
}

function englishVoice() {
  const voices = window.speechSynthesis.getVoices();
  const american = voices.find((voice) => (voice.lang || "").toLowerCase().startsWith("en-us"));
  if (american) {
    return american;
  }
  const british = voices.find((voice) => (voice.lang || "").toLowerCase().startsWith("en-gb"));
  if (british) {
    return british;
  }
  return voices.find((voice) => (voice.lang || "").toLowerCase().startsWith("en")) || null;
}

function bindUtterance(utterance) {
  const voice = englishVoice();
  if (voice) {
    utterance.voice = voice;
  }
  utterance.lang = "en-US";
  utterance.rate = 0.92;
  utterance.pitch = 1.0;
}

export function unlockDerja() {
  if (!("speechSynthesis" in window)) {
    return;
  }
  window.speechSynthesis.resume();
  window.speechSynthesis.getVoices();
  if (speechUnlocked) {
    return;
  }
  const primer = new SpeechSynthesisUtterance(" ");
  primer.volume = 0.01;
  bindUtterance(primer);
  window.speechSynthesis.speak(primer);
  speechUnlocked = true;
}

function speakNow(text) {
  const utterance = new SpeechSynthesisUtterance(text);
  bindUtterance(utterance);
  window.speechSynthesis.resume();
  window.speechSynthesis.cancel();
  window.setTimeout(() => {
    window.speechSynthesis.resume();
    window.speechSynthesis.speak(utterance);
  }, 80);
}

export function speakDerja(text) {
  const spoken = englishSpeechText(text);
  if (!spoken) {
    return;
  }
  if (!("speechSynthesis" in window)) {
    return;
  }

  if (window.speechSynthesis.getVoices().length === 0) {
    let started = false;
    const start = () => {
      if (started) {
        return;
      }
      started = true;
      window.speechSynthesis.removeEventListener("voiceschanged", start);
      speakNow(spoken);
    };
    window.speechSynthesis.addEventListener("voiceschanged", start);
    window.setTimeout(start, 400);
    return;
  }

  speakNow(spoken);
}

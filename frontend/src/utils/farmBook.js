const STORAGE_KEY = "haran-farm-book";

export function readFarmChecks() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function saveFarmCheck(entry) {
  const checks = readFarmChecks();
  const checkedAt = new Date().toISOString();
  const days = Number(entry.checkAgainDays) || 7;
  const checkAgain = new Date(Date.now() + days * 24 * 60 * 60 * 1000).toISOString();
  const record = {
    id: `${Date.now()}`,
    checkedAt,
    checkAgain,
    lat: entry.lat,
    lng: entry.lng,
    status: entry.status,
    disease: entry.disease,
    severity: entry.severity,
    solution: entry.solution,
    damagePercent: entry.damagePercent ?? null,
  };
  const next = [record, ...checks].slice(0, 40);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  return next;
}

export function formatWhen(iso) {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) {
    return "";
  }
  return date.toLocaleString(undefined, {
    day: "numeric",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function formatPlace(lat, lng) {
  const latitude = Number(lat);
  const longitude = Number(lng);
  if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) {
    return "Location not set";
  }
  return `${latitude.toFixed(5)}, ${longitude.toFixed(5)}`;
}

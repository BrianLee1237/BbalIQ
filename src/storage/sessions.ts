import AsyncStorage from "@react-native-async-storage/async-storage";

import type { SessionRecord } from "../../types/tracking";

const SESSION_STORAGE_KEY = "@activetrack/sessions/v1";
const MAX_SAVED_SESSIONS = 50;

export async function loadSessions(): Promise<SessionRecord[]> {
  const value = await AsyncStorage.getItem(SESSION_STORAGE_KEY);
  if (!value) return [];

  try {
    const parsed: unknown = JSON.parse(value);
    return Array.isArray(parsed) ? (parsed as SessionRecord[]) : [];
  } catch {
    return [];
  }
}

export async function saveSession(session: SessionRecord): Promise<SessionRecord[]> {
  const existing = await loadSessions();
  const next = [session, ...existing.filter((item) => item.id !== session.id)].slice(
    0,
    MAX_SAVED_SESSIONS,
  );
  await AsyncStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(next));
  return next;
}

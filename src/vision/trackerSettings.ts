export const MIN_AUTOMATIC_HOOP_CONFIDENCE = 0.18;

export const HOOP_TRACKER_SETTINGS = {
  track_high_thresh: 0.08,
  track_low_thresh: 0.025,
  new_track_thresh: 0.08,
  track_buffer: 36,
  match_thresh: 0.78,
  fuse_score: true,
};

export const BALL_TRACKER_SETTINGS = {
  track_high_thresh: 0.18,
  track_low_thresh: 0.03,
  new_track_thresh: 0.16,
  track_buffer: 18,
  match_thresh: 0.82,
  fuse_score: true,
};

export const PLAYER_TRACKER_SETTINGS = {
  track_high_thresh: 0.32,
  track_low_thresh: 0.08,
  new_track_thresh: 0.3,
  track_buffer: 45,
  match_thresh: 0.8,
  fuse_score: true,
};

# VigiLense presentation video: narration script

Length: 6:47 · 1080p · 23 scenes. Subtitles: [VigiLense-presentation.srt](VigiLense-presentation.srt).

## 0:00 · Cold open

Somewhere right now, a security operator is watching forty camera feeds. Their attention can be on only one. And in the few seconds they look away, a weapon appears, smoke begins to rise, or a person of interest slips out of view.

## 0:16 · Title

This is VigiLense: an AI surveillance and incident response system that watches every feed, understands what it sees, and brings the most critical moment straight to the operator.

## 0:30 · The problem

Conventional CCTV depends almost entirely on human attention. In universities, shopping malls, hospitals, airports, banks and railway stations, operators must observe many screens at once. The result is fatigue, missed incidents, delayed response, and poor situational awareness.

## 0:51 · The problem (continued)

These systems are passive. They record and display, but they cannot recognise a weapon, fire or smoke. A critical event can unfold on a feed that is not on the main screen. And when a person moves from one camera to another, tracking ends, especially when their face is hidden, covered, or turned away.

## 1:12 · Objectives

VigiLense answers this with clear objectives. Build a real-time prototype with Ultralytics YOLO26, OpenCV and Python. Monitor at least three camera feeds. Detect persons, weapons, fire and smoke. Track people within each camera, and follow a selected person across cameras with a global ID. Then prioritise events, switch the most important feed to the dashboard, notify the operator, and log every incident with evidence.

## 1:43 · Input and preprocessing

Here is how it works. Each camera source, whether CCTV, IP, RTSP, webcam or recorded video, is registered with an ID and a location. OpenCV captures each stream, extracts frames, and resizes, normalises and enhances them for analysis.

## 2:04 · YOLO26 detection

YOLO26 is the main detection model. For every frame it returns object classes, bounding boxes and confidence scores, for persons, weapon-like objects, fire and smoke. Pretrained models are fine-tuned on public, authorised and controlled datasets.

## 2:22 · Local tracking

Next, ByteTrack or BoT-SORT gives every person a local tracking ID inside each camera, keeping identities stable from frame to frame, even in a crowd.

## 2:34 · Person re-identification

When an operator marks a person of interest, VigiLense extracts their appearance: clothing, body shape, helmet, bag, and other accessories. An OSNet re-identification model, through Torchreid, compares these features against people seen by other cameras.

## 2:53 · Gait and global ID

When enough walking frames are available, OpenGait adds gait and silhouette features. Appearance, gait, camera location, timestamp and expected transition time are fused into a single global person ID, with a camera trail and last known location. No face is required.

## 3:13 · Decision engine

Every detection becomes an event. The decision engine scores it by event type, confidence, persistence and camera location, so a weapon, a fire, or the reappearance of a marked person rises above routine activity.

## 3:29 · Operator dashboard

The highest priority feed is switched automatically to the primary view of a custom web dashboard, alongside the multi-camera grid, live notifications, the person trail, and system status.

## 3:42 · Alerts and incident log

Browser alerts and optional warning sounds notify the operator instantly. Snapshots and short evidence clips are saved, and every incident is logged with its ID, time, camera, event type, confidence, priority, global person ID, evidence link, and review status.

## 4:02 · System architecture

Together, this forms a six-layer real-time architecture: video input, stream ingestion, the AI processing engine, the decision and event engine, a FastAPI backend with MongoDB or PostgreSQL, and a React and Tailwind dashboard.

## 4:20 · Monitoring loop

Every frame from every camera follows one monitoring loop: read, preprocess, detect, track, match, score, switch, notify, save, and log. Operators can then review each incident, and confirm it or mark it as a false alarm.

## 4:39 · Evaluation plan

The system will be evaluated for detection accuracy, false alarms, tracking continuity, cross-camera matching, ID switches, processing speed, latency, dashboard response, camera switching, and logging correctness.

## 4:55 · Existing CCTV vs VigiLense

Compared with conventional CCTV, the difference is clear. Manual observation becomes AI monitoring. Passive display becomes active detection. Single-camera tracking becomes cross-camera identity. And manual evidence search becomes automatic, organised incident records.

## 5:14 · Applications

VigiLense can serve universities, airports and transport hubs, shopping malls, hospitals, offices, banks, warehouses, and high-security facilities, with research value in detection, multi-object tracking, re-identification and gait recognition.

## 5:32 · Tools and hardware

It is built with Python, PyTorch, YOLO26, OpenCV, ByteTrack, Torchreid and OpenGait for intelligence; FastAPI, React, Tailwind and a database for the application; running on an RTX-class GPU with at least three cameras.

## 5:51 · Plan and budget

The work is planned across eight months, from requirements and data collection, through training, tracking, re-identification and the dashboard, to integration, testing and the final demonstration, within an estimated budget of one hundred thousand rupees.

## 6:09 · Future enhancements

Looking ahead, VigiLense can grow with ethically approved facial recognition, licence plate recognition, a mobile alert app, edge and cloud deployment, role-based access, encrypted evidence, and integration with emergency response systems.

## 6:26 · The team

VigiLense is a Computer Engineering final year project by Hassnain Khan, Khurram Ali Khan and Yasir Hassan, under the supervision of Doctor Syed Khawar Hussain Shah.

## 6:38 · Outro

VigiLense. Watching every feed, so nothing is missed. Thank you.

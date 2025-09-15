# Project Roadmap

This document outlines the planned features and development direction for the Live AI Camera Monitoring Suite.

## Phase 1: Core Functionality (Complete)

*   [x] Connect to a single IP camera stream.
*   [x] Perform real-time object detection using YOLOv3.
*   [x] Display the processed video feed in a local window.
*   [x] Basic setup and usage instructions in `README.md`.

## Phase 2: Web Application and Multi-Camera Support (Complete)

This phase focuses on transforming the application into a web-based service capable of handling multiple cameras.

*   [x] **Web Server Interface:**
    *   Implement a web server using Flask to provide a web-based UI.
    *   Stream processed video feeds to the browser.
*   [x] **Multi-Camera Support:**
    *   Refactor the codebase to support multiple camera streams simultaneously.
    *   Allow configuration of multiple camera URLs.
*   [x] **Notification System:**
    *   Implement a basic notification system (e.g., logging to a file) when specific objects are detected.
*   [x] **Documentation Website:**
    *   Set up a documentation website using GitHub Pages.

## Phase 3: Advanced Features (Future)

This phase will focus on enhancing the application with more advanced features.

*   [ ] **User Authentication:**
    *   Add a user login system to control access to the web interface.
*   [ ] **Advanced Notifications:**
    *   Integrate with services like Email, Slack, or Pushbullet for more advanced notifications.
*   [ ] **Event History and Recording:**
    *   Implement a system to record video clips when events of interest occur.
    *   Provide a web interface to view past events and recordings.
*   [ ] **Configuration via UI:**
    *   Allow users to add, remove, and configure cameras directly from the web interface, instead of editing a configuration file.
*   [ ] **Improved AI Models:**
    *   Allow swapping different object detection models (e.g., YOLOv4, SSD).
    *   Explore other AI tasks like face recognition or license plate recognition.

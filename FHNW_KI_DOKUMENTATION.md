# FHNW-konforme Zusammenfassung der KI-Nutzung (gemäss Skill-Standard)

## 1. Einordnung der Nutzungsart
**Nutzungsart: Methodische Nutzung (Regelfall)**

Begründung:
1. KI wurde als Entwicklungs- und Unterstützungstool für Konzeption, Implementierung, Debugging, Absicherung und Dokumentation eingesetzt.
2. Es wurden keine KI-Texte oder KI-Inhalte als wissenschaftliche Quelle unverändert übernommen.
3. Die Ergebnisse (Code, Konfiguration, Sicherheitsmassnahmen) wurden iterativ geprüft, angepasst und im Projektkontext umgesetzt.

---

## 2. Dokumentationstabelle (für Methodenteil/Anhang)

| Datum | Tool / Modell | Aufgabe | Prompt (Kurzfassung) | Ergebnis / Ausgabe | Eigenleistung / Überarbeitung | Begründung des Einsatzes |
|---|---|---|---|---|---|---|
| 01.04.2026 | GitHub Copilot Chat (GPT-5.3-Codex) | Projektaufbau React/FastAPI | "Baue DLSS-Harness-App mit Fixed vs Meta-Harness" | Initiale App-Struktur, Endpoints, UI | Architekturentscheidungen, lokale Ausführung, manuelle Tests | Schneller Start mit lauffähigem Prototyp |
| 01.04.2026 | GitHub Copilot Chat (GPT-5.3-Codex) | Plattform-Fix Windows ARM64 | "Installationsfehler beheben (numpy/opencv/uvicorn)" | Dependency-Anpassungen, Startfixes | Umgebungsspezifische Anpassungen/Neustarts | Technische Kompatibilität sicherstellen |
| 01.04.2026 | GitHub Copilot Chat (GPT-5.3-Codex) | Qualitätsverbesserung Meta-Harness | "Warum schlechte/gleiche Ergebnisse? verbessern" | Parameter-Tuning, Best-Iteration-Logik | Bewertung der Metriken, Entscheidung über Iterationsauswahl | Stabilere Ergebnisse und nachvollziehbare Adaptation |
| 01.–02.04.2026 | GitHub Copilot Chat (GPT-5.3-Codex) | Externes Model-API integrieren | "Model-API Harness mit Iterationen einbauen" | Dritter Modus inkl. API-Loop | Auswahl/Anpassung von Endpoints und Defaults | Vergleich lokaler vs. externer Verbesserungsstrategie |
| 02.04.2026 | GitHub Copilot Chat (GPT-5.3-Codex) | Google-Gemini-Integration | "Direkt mit Google lauffähig machen" | Google-Request/Response-Mapping, Fehlermeldungen verbessert | Modell- und Quota-Prüfung, Validierung per Testläufen | Praktische Nutzbarkeit mit vorhandenem API-Setup |
| 02.04.2026 | GitHub Copilot Chat (GPT-5.3-Codex) | Security/Robustness gemäss Review | "4 Findings umsetzen" | CORS-Härtung, Upload-Limits, Memory-Bounds, Repo-Hygiene, Report | Priorisierung, Relevanzprüfung, Konfigurationsfestlegung | Stabilität und Sicherheit für lokalen Betrieb erhöhen |

---

## 3. Methodenteil-Text (kopierfähig, FHNW-Stil)

Für die Entwicklung und Absicherung der Anwendung wurde ein generatives KI-System (GitHub Copilot Chat, Modell: GPT-5.3-Codex, 2026) als methodisches Hilfsmittel eingesetzt. Die KI wurde zur Strukturierung der Implementierung, für Codevorschläge, zur Fehlerdiagnose sowie zur Erstellung technischer Dokumentation verwendet. Die generierten Ausgaben dienten als Ausgangspunkt; alle vorgeschlagenen Änderungen wurden im Projektkontext manuell geprüft, angepasst und durch lokale Tests validiert (Build-, Laufzeit- und Endpoint-Tests).
Die wissenschaftliche bzw. fachliche Verantwortung für Architektur, Priorisierung der Review-Findings, Sicherheitsentscheidungen und finale Umsetzung lag vollständig bei der Autorin/dem Autor.

---

## 4. Integritätscheck (FHNW-orientiert)

Durchgeführt bzw. empfohlen:
1. **Nicht delegiert:** Fachliche Entscheidungen (Relevanzbewertung der Findings, Umsetzungsprioritäten) wurden nicht an KI delegiert.
2. **Geprüft:** KI-Ausgaben wurden mittels Build/Compile/Runtime-Checks, API-Tests und manueller Review verifiziert.
3. **Dokumentiert:** KI-Einsatz ist als methodische Nutzung transparent beschrieben.
4. **Risiken beachtet:** Fehlinterpretationen bei Security-Themen wurden durch explizite Relevanzprüfung und konkrete Gegenmassnahmen reduziert.
5. **Datenschutz:** Zugangsdaten/Secrets gehören nicht in öffentliche Dokumentation; Schlüsselrotation wird empfohlen, falls ein Key offengelegt wurde.

---

## 5. Deklarationshinweis
Nach FHNW-Logik ist dies **methodische Nutzung**.
Damit ist primär erforderlich:
1. Dokumentationstabelle (wie oben),
2. transparenter Methodenteiltext,
3. reflektierte Begründung von Zweck, Umfang und Kontrolle der KI-Nutzung.

Eine Zitierung von KI als wissenschaftliche Quelle im Fliesstext ist hier nicht nötig, solange keine direkte Übernahme von KI-Inhalten als Quelle erfolgt.

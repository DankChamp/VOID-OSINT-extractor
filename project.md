# VIDEO INTELLIGENCE
## Project Specification — Transcript-First, Multi-AI Knowledge Extraction

**Project type:** Python application  
**Initial interface:** CLI  
**Architecture:** Modular, multi-provider, pipeline-based  
**Primary objective:** Extract, consolidate, and structure the information actually presented in YouTube videos.

---

# 1. Core Mission

Build a program that accepts a YouTube URL, obtains its transcript, and transforms the information contained in that transcript into a comprehensive, structured knowledge dataset.

The system must support multiple AI providers and configurable processing pipelines.

**This is not a conventional AI summarizer.**

The application must not simply send a URL or transcript to one model, ask for a summary, and return whatever that model produces.

Instead, it must:

1. Obtain the original transcript and preserve it.
2. Divide the transcript into manageable, timestamped segments.
3. Send transcript segments to configurable AI providers for structured information extraction.
4. Collect all extracted datasets.
5. Merge complementary information and eliminate genuine duplicates.
6. Verify extracted claims against the original transcript.
7. Identify omissions, uncertainty, and conflicting interpretations.
8. Build one comprehensive, hierarchical knowledge dataset.
9. Generate detailed notes, presentation outlines, and concise summaries from that dataset.

The output must preserve as much unique, meaningful information from the video as possible without unnecessary repetition.

## Fundamental principle

**The transcript is the primary source of truth. AI models are processing tools, not independent sources of facts about the video.**

If a model produces information that cannot be supported by the transcript, the application must not present it as something the speaker said.

---

# 2. Expected User Workflow

The intended experience is:

```text
User pastes YouTube URL
          |
          v
Retrieve video metadata
          |
          v
Obtain transcript with timestamps
          |
          v
Save original transcript
          |
          v
Clean transcript without losing meaning
          |
          v
Divide transcript into overlapping chunks
          |
          v
Send chunks to configured AI pipelines
          |
          v
Collect all model outputs
          |
          v
Normalize and validate extracted data
          |
          v
Merge complementary information
          |
          v
Identify and remove genuine duplicates
          |
          v
Verify claims against transcript evidence
          |
          v
Construct hierarchical knowledge dataset
          |
          v
Generate requested outputs
          |
          +---- Detailed notes
          +---- PPT outline
          +---- Quick summary
          +---- Structured JSON
```

The user should be able to configure which models handle individual stages and how many extraction passes are performed.

---

# 3. Critical Requirement: Preserve the Original Information

The application must never discard the original transcript after processing.

Maintain separate representations:

- `raw_transcript`: the transcript as acquired.
- `normalized_transcript`: cleaned text with source timing preserved.
- `chunks`: timestamped sections prepared for processing.
- `extractions`: raw outputs returned by each AI pipeline.
- `verified_items`: extracted items checked against transcript evidence.
- `knowledge_dataset`: the consolidated, structured result.
- `outputs`: generated notes and presentation materials.

Cleaning must not change the meaning of the speaker's statements.

Do not silently delete repeated passages from the source transcript. Repetition may contain additional context or explain a concept differently.

The system should preserve the source and consolidate repetition only in the final knowledge representation.

---

# 4. Transcript Acquisition

Implement a dedicated ingestion subsystem.

It should:

- Accept a YouTube URL.
- Validate the URL and identify the video.
- Retrieve available metadata, including title and duration.
- Obtain available captions or transcripts.
- Preserve segment timestamps.
- Detect transcript language where possible.
- Report whether the transcript is manually created or automatically generated when this information is available.
- Save the transcript locally.
- Handle missing captions, unavailable videos, and retrieval failures.

Do not assume that every video has a transcript.

If no usable transcript is available, report the problem and offer an optional speech-to-text transcription path for audio the user is authorized to process.

Do not fabricate transcript content or silently substitute an unrelated source.

The original video should remain the reference for resolving ambiguities.

---

# 5. Timestamp and Provenance System

Every extracted knowledge item must be traceable to its source.

Represent transcript segments using a structure such as:

```json
{
  "segment_id": "segment_001",
  "start_seconds": 120.0,
  "end_seconds": 145.0,
  "text": "Transcript text goes here."
}
```

Each extracted item should retain:

- Unique item ID.
- Item category.
- Extracted statement.
- Supporting transcript segment IDs.
- Start and end timestamps.
- Extraction pipeline ID.
- Model/provider ID.
- Verification status.
- Any uncertainty or conflict flags.

A knowledge item supported by multiple transcript passages should reference all relevant passages.

Timestamps must come from actual transcript timing or verified source locations. Models must not invent them.

---

# 6. Intelligent Transcript Chunking

Create a chunking subsystem that prepares transcripts for AI processing.

Requirements:

- Respect sentence boundaries where possible.
- Use configurable token or character limits.
- Preserve start and end timestamps.
- Use a configurable overlap between neighboring chunks.
- Keep enough surrounding context to understand references.
- Avoid splitting definitions, lists, and explanations unnecessarily.
- Support long videos without requiring the entire transcript to fit into one model context.

Every chunk should have a stable ID and links to its source segments.

Chunking must be deterministic when given the same configuration and transcript.

---

# 7. Multi-AI Provider Architecture

Build a provider abstraction so models can be replaced without rewriting the application.

Initial integration targets:

- Ollama for local models.
- DeepSeek.
- OpenAI-compatible API providers.
- Groq.
- Google Gemini or other providers through dedicated adapters when implemented.

**NotebookLM integration must be treated separately from ordinary API providers.**

Do not assume NotebookLM has a supported public API for arbitrary automated notebook creation or result retrieval. Verify the current supported integration options before implementing it.

If automated integration is unavailable, implement a clean NotebookLM handoff workflow:

1. Prepare source-grounded extraction packets.
2. Allow the user to import supported source material into NotebookLM.
3. Provide structured prompts and export instructions.
4. Import the resulting structured data back into the application.
5. Validate and merge it with the other extraction results.

Never pretend that a manual handoff is a fully automated integration.

Use official, supported interfaces wherever possible. Do not automate private endpoints or circumvent platform restrictions.

---

# 8. Configurable AI Pipelines

Each pipeline should have a distinct responsibility.

Initial pipeline categories:

### A. Concept Extraction

Extract major topics, minor topics, subtopics, relationships, and explanations.

### B. Definition Extraction

Extract terminology, definitions, distinctions, and technical vocabulary.

### C. Fact and Data Extraction

Extract numbers, statistics, dates, measurements, specifications, and factual statements made by the speaker.

### D. Example Extraction

Extract demonstrations, hypothetical scenarios, case studies, analogies, and real-world examples.

### E. Framework Extraction

Extract named frameworks, standards, methodologies, classifications, and models discussed in the transcript.

### F. Process Extraction

Extract procedures, workflows, ordered steps, and decision processes.

### G. Comparison Extraction

Extract comparisons, differences, advantages, disadvantages, limitations, and trade-offs.

### H. Coverage Audit

Inspect transcript chunks and compare them with extracted items to identify potentially missed definitions, topics, examples, claims, and other meaningful information.

This audit must work from the source transcript rather than relying solely on what other models already extracted.

The user must be able to configure which pipelines run and which models perform each task.

---

# 9. AI Output Contract

Models must return structured data rather than unrestricted summaries wherever possible.

Example:

```json
{
  "items": [
    {
      "type": "definition",
      "statement": "Definition supported by the transcript.",
      "details": [
        "First distinct detail.",
        "Second distinct detail."
      ],
      "source_segments": [
        "segment_001"
      ],
      "uncertainties": []
    }
  ]
}
```

Validate every response against application-defined schemas.

If the response is invalid:

1. Attempt a controlled repair or retry.
2. Validate it again.
3. Record the failure if validation still fails.
4. Continue other independent tasks when possible.

Store raw model responses separately from normalized extraction records.

Do not let an invalid model response corrupt the knowledge dataset.

---

# 10. Evidence Verification

The application must distinguish between information extracted by an AI and information actually supported by the transcript.

Use explicit verification states:

- `supported`: the cited transcript passage supports the claim.
- `partially_supported`: some details are supported, but others need review.
- `unsupported`: the available source does not support the claim.
- `needs_review`: the evidence is ambiguous or verification failed.

The verifier should compare each item's statement with its cited transcript passages.

For high-value items, it may inspect additional surrounding context.

Where practical, combine deterministic checks with a separate verification model. Do not rely exclusively on the same model that generated the claim.

Unsupported additions must not be silently included as video-derived facts.

If external research is introduced in a future version, keep it in a separate dataset with explicit external citations. Never mix external facts into the transcript-derived content without identifying their origin.

---

# 11. Deduplication and Information Consolidation

The goal is to remove repetition while retaining every distinct piece of information.

Deduplication must distinguish:

1. Exact duplicates.
2. Semantic duplicates.
3. Complementary details.
4. Related but distinct concepts.
5. Contradictory claims.

Example:

Original extractions:

- Risk assessment identifies potential threats.
- Risk assessment evaluates the likelihood of threats.
- Risk assessment evaluates potential impact.

Consolidated result:

**Risk Assessment**

- Identifies potential threats.
- Evaluates their likelihood.
- Evaluates their potential impact.

All supporting timestamps must be preserved.

Do not merge two statements simply because they discuss the same topic.

Do not discard an item merely because only one model extracted it.

A unique item should be retained if it is supported by the source, even if every other model missed it.

Deduplication must be explainable and reversible wherever practical.

---

# 12. Conflict Detection

Different models may extract inconsistent statements or interpret the same passage differently.

The application must:

- Identify conflicting claims.
- Retain the original extracted statements.
- Compare their supporting evidence.
- Resolve the conflict only when the source supports a clear resolution.
- Otherwise, flag the issue for review.

Do not use majority voting as the sole mechanism for deciding factual correctness.

Example:

```json
{
  "type": "conflict",
  "claim_ids": [
    "item_012",
    "item_047"
  ],
  "status": "needs_review",
  "resolution": null
}
```

If the transcript is unclear, preserve that uncertainty rather than inventing certainty.

---

# 13. Final Knowledge Dataset

All validated and consolidated items must be organized into a hierarchical knowledge structure.

Example:

```text
Cybersecurity Risk Management
|
+-- Risk Fundamentals
|   +-- Definitions
|   +-- Risk Components
|   +-- Importance
|
+-- Risk Assessment
|   +-- Asset Identification
|   +-- Threat Identification
|   +-- Vulnerability Assessment
|   +-- Risk Analysis
|
+-- Risk Treatment
|   +-- Mitigation
|   +-- Transfer
|   +-- Acceptance
|   +-- Avoidance
|
+-- Frameworks
+-- Processes
+-- Examples
+-- Important Claims
```

This hierarchy is illustrative only. Generate the actual structure from the information found in each video.

Do not insert topics from this example into unrelated videos.

The dataset must preserve:

- Parent-child topic relationships.
- Unique knowledge items.
- Evidence references.
- Timestamps.
- Verification status.
- Cross-references between related concepts.
- Conflict records.
- Extraction metadata.

Use a versioned JSON schema so future versions can extend the dataset without breaking older projects.

---

# 14. Output Generation

All outputs must be generated from the consolidated knowledge dataset, not independently summarized from the original transcript.

## A. Comprehensive Notes

Generate a Markdown document containing:

- Hierarchical topics and subtopics.
- Definitions and explanations.
- Distinct facts and statistics.
- Examples and demonstrations.
- Processes and frameworks.
- Comparisons and limitations.
- Important claims.
- Source timestamps.
- Unresolved conflicts and uncertainty.

Prioritize coverage over brevity.

## B. PowerPoint Outline

Generate an outline suitable for creating a presentation of 10 or more slides when the video contains sufficient material.

Each slide should contain:

- Slide title.
- Learning objective or main message.
- Key points.
- Supporting facts and definitions.
- Relevant examples.
- Source timestamps.
- Optional speaker notes.

Group related concepts logically and avoid repeating the same content across slides.

Do not invent information just to reach a target slide count. If a video lacks enough material for ten meaningful slides, report that limitation.

## C. Quick Summary

Generate a concise overview only after the detailed dataset exists.

The quick summary is a secondary output and must never replace comprehensive extraction.

## D. Structured JSON

Export the complete knowledge dataset, including provenance, verification status, and conflict records.

---

# 15. Project Storage

Use SQLite and the local filesystem for the MVP.

Organize persistent data by project and video.

Example:

```text
data/
  projects/
    project_id/
      video_id/
        metadata.json
        raw_transcript.json
        normalized_transcript.json
        chunks.json
        raw_extractions/
        verified_items.json
        knowledge.json
        conflicts.json
        outputs/
          full_notes.md
          ppt_outline.md
          summary.md
```

Use stable IDs, safe file operations, and atomic writes where practical.

Cache completed stages so that changing an output format does not require reprocessing the entire video.

If a pipeline configuration changes, invalidate only the affected processing stages.

---

# 16. CLI Interface

Build a CLI-first application.

The initial command should resemble:

```bash
video-intel analyze "YOUTUBE_URL" --mode deep
```

Additional options:

```bash
video-intel analyze "YOUTUBE_URL" --mode ppt
video-intel analyze "YOUTUBE_URL" --mode quick
video-intel analyze "YOUTUBE_URL" --provider ollama
video-intel analyze "YOUTUBE_URL" --pipeline maximum-coverage
video-intel providers
video-intel pipelines
video-intel inspect VIDEO_ID
```

The application must show processing progress, failed stages, and output locations.

Support resuming interrupted jobs wherever practical.

---

# 17. Configuration and Security

Keep API keys in environment variables or an appropriate local secrets mechanism.

Provide a `.env.example` containing placeholder values only.

Never:

- Hard-code API keys.
- Commit real credentials.
- Print secrets in logs.
- Send the original transcript to a cloud provider unless that provider is selected for the relevant pipeline.
- Silently change provider configuration.
- Claim a task was completed when a provider failed.

Allow the user to select local-only, cloud-only, or hybrid execution.

Provider selection must be explicit and inspectable.

---

# 18. Recommended Technology

Use Python for the initial implementation.

Prefer:

- Pydantic for structured data validation.
- A standard asynchronous HTTP client for API requests.
- SQLite for metadata and processing state.
- YAML for pipeline configuration.
- A mature CLI library.
- Ollama's supported API for local inference.
- Provider-specific adapters for cloud models.
- Pytest for automated testing.

Choose additional dependencies only when they solve a concrete problem.

Do not add a vector database, complex agent framework, or elaborate GUI unless the implementation demonstrates a clear need.

Verify current library interfaces before relying on them.

---

# 19. Error Handling and Reliability

Handle:

- Invalid YouTube URLs.
- Missing or inaccessible transcripts.
- Transcription failures.
- API timeouts and rate limits.
- Provider outages.
- Invalid model responses.
- Malformed JSON.
- Chunk-processing failures.
- Duplicate processing jobs.
- Interrupted runs.
- Partial provider failure.

Independent pipelines should continue when safe.

If one provider fails, retain completed results and report which parts remain incomplete.

Never label a partially processed dataset as fully verified.

---

# 20. Testing Requirements

Write automated tests for:

- Transcript ingestion and normalization.
- Timestamp preservation.
- Chunk boundaries and overlap.
- Schema validation.
- Provider abstraction.
- Extraction parsing.
- Semantic deduplication.
- Complementary information merging.
- Evidence verification.
- Conflict detection.
- Knowledge hierarchy generation.
- Markdown and JSON output.
- Interrupted-job recovery.

Use synthetic fixture transcripts and mocked AI providers for unit tests.

Real API calls must not be required to run the basic test suite.

Include tests proving that:

1. A unique source-supported fact is not removed because only one model extracted it.
2. Unsupported model-generated claims are flagged.
3. Repeated explanations are consolidated without losing distinct details.
4. Multiple source timestamps survive consolidation.
5. Contradictory claims remain unresolved when the transcript cannot resolve them.
6. Changing an output format does not trigger unnecessary AI calls.

---

# 21. Implementation Phases

Implement incrementally.

## Phase 1 — Foundation

- Inspect the repository.
- Establish project structure.
- Define schemas and data models.
- Implement configuration, logging, and CLI.
- Add unit-test infrastructure.

## Phase 2 — Transcript Ingestion

- YouTube metadata.
- Transcript acquisition.
- Timestamp preservation.
- Transcript storage.
- Cleaning and normalization.

## Phase 3 — Chunking

- Configurable chunk size.
- Timestamped segments.
- Context overlap.
- Persistent chunk storage.

## Phase 4 — Provider Infrastructure

- Provider interface.
- Ollama adapter.
- One cloud-provider adapter.
- Configuration-driven model registry.
- Error handling and retries.

## Phase 5 — Extraction Pipelines

- Concepts.
- Definitions.
- Facts and data.
- Examples.
- Frameworks.
- Processes.
- Comparisons.
- Coverage audit.

## Phase 6 — Aggregation and Verification

- Schema normalization.
- Deduplication.
- Complementary information merging.
- Evidence verification.
- Conflict detection.
- Hierarchical dataset generation.

## Phase 7 — Outputs

- Comprehensive Markdown notes.
- Structured JSON.
- PPT outline.
- Quick summary.

## Phase 8 — NotebookLM Workflow

- Prepare source packets.
- Generate structured extraction prompts.
- Implement supported import/export handoff.
- Add automated integration only if a supported interface is available.
- Validate and merge imported results.

## Phase 9 — Reliability

- Resume support.
- Caching.
- Logging improvements.
- Integration tests.
- Documentation.

Do not begin GUI development until the CLI pipeline works reliably.

---

# 22. Definition of Done

The MVP is complete when the following command produces a usable result:

```bash
video-intel analyze "YOUTUBE_URL" --mode deep
```

Expected outputs:

```text
full_notes.md
knowledge.json
ppt_outline.md
summary.md
```

The application must demonstrably:

- Use the actual video transcript as its primary evidence.
- Preserve timestamps and source provenance.
- Process transcript chunks through configurable extraction pipelines.
- Support multiple AI providers.
- Merge results without discarding unique supported information.
- Remove genuine redundancy.
- Flag unsupported claims and unresolved conflicts.
- Generate all outputs from the same consolidated dataset.
- Recover gracefully from partial provider failures.
- Avoid claiming integrations or verification that have not actually occurred.

---

# 23. Instructions to the Coding Agent

You are responsible for implementing this specification in the existing repository.

Before coding:

1. Inspect the current repository and installed dependencies.
2. Identify existing functionality that can be reused.
3. Check the current project state and available hardware.
4. Create a short implementation plan.
5. Begin with Phase 1 only.

After Phase 1 is tested, continue incrementally through the remaining phases.

Do not generate the entire application in one uncontrolled code dump.

Do not create placeholder integrations that pretend to work.

Do not silently substitute ordinary summarization for structured information extraction.

Do not assume that NotebookLM offers a supported automation API.

When a requirement cannot yet be implemented, document the limitation and provide the next concrete implementation step.

Keep modules small, typed, testable, and independent wherever practical.

**The priority is a reliable, source-grounded information extraction engine—not a flashy interface.**

The final product should turn a video transcript into a reusable knowledge dataset from which many different outputs can be generated without losing the original information.

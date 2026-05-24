# pairwise-voting Specification

## Purpose
Rank images through comparative A/B pairwise voting using Elo or Condorcet scoring algorithms,
producing bias-reduced aesthetic rankings from a series of random image duels via a local
Gradio interface.

---

## Requirements

### Requirement: Pairwise Duel Presentation
The system SHALL present two randomly selected images side by side and allow the user to
select the aesthetically preferred one.

#### Scenario: Random duel drawn
- **GIVEN** a pool of images not yet definitively ranked
- **WHEN** the interface loads or after a vote is cast
- **THEN** two distinct images are selected at random and displayed side by side

#### Scenario: Condorcet — pool exhaustion
- **GIVEN** all remaining images have reached score 0 or 5 in Condorcet mode
- **WHEN** a new duel is requested
- **THEN** the interface displays a completion message indicating all images are ranked

---

### Requirement: Elo Scoring
The system SHALL update image scores using the Elo algorithm (K=32) after each duel,
reflecting the relative skill levels of the competing images.

#### Scenario: Winner/loser score update (Elo)
- **GIVEN** two images with existing Elo scores
- **WHEN** the user selects the preferred image
- **THEN** the winner's score increases and the loser's decreases according to the Elo formula with K=32

#### Scenario: Final score normalization
- **GIVEN** a session ends with raw Elo scores
- **WHEN** results are exported
- **THEN** all scores are linearly normalized to the [0.0, 5.0] range before CSV export

---

### Requirement: Condorcet Scoring
The system SHALL update image scores linearly (+0.1 / -0.1, bounded [0, 5]) after each duel
and permanently exclude images reaching the extremes from future draws.

#### Scenario: Score boundary enforcement
- **GIVEN** an image reaches a score of 5.0 or 0.0 in Condorcet mode
- **WHEN** the next duel is drawn
- **THEN** the image is excluded from further comparisons

---

### Requirement: Session Persistence
The system SHALL automatically save the voting state after each duel and reload it on restart,
enabling sessions to be interrupted and resumed without data loss.

#### Scenario: Session resumed after interruption
- **GIVEN** `annotation_state.json` exists from a previous session
- **WHEN** `ab_vote.py` is relaunched
- **THEN** all previous scores are restored and voting continues from where it was left off

---

### Requirement: Results Export
The system SHALL export final rankings and duel history to a CSV file upon session completion.

#### Scenario: CSV export
- **GIVEN** a voting session is complete or the user requests export
- **WHEN** results are saved
- **THEN** a CSV is written to `data/outputs/predictions/pairwise_rankings.csv` containing
  image paths and final scores

# Task Plan: Restructure Transformer Project for Modularity

## Goal
Reorganize the project into a clean, modular structure that separates training, experimentation, API serving, and data management, while ensuring all training and fine-tuning scripts remain fully functional.

## Phases
- [x] Phase 1: Analyze current structure and dependencies
- [x] Phase 2: Design new folder structure
- [x] Phase 3: Create new directories and move files
- [x] Phase 4: Update import paths and configurations
- [ ] Phase 5: Test training scripts functionality
- [ ] Phase 6: Test API functionality
- [ ] Phase 7: Update documentation

## Status
**Currently in Phase 5** - Testing training scripts functionality

## Key Questions
1. What are the main components: training, inference, API, data, models, experiments?
2. Which files are core vs experimental?
3. How to handle multiple model versions (model.py, model_2.py)?
4. Where to put notebooks - keep in root or separate?
5. How to make training scripts easily runnable?

## Decisions Made
- Separate src/ for core code, experiments/ for notebooks and scripts, data/ for datasets, models/ for checkpoints
- Keep training scripts in experiments/ but ensure they can run from there
- Use relative imports where possible
- Maintain backward compatibility for existing paths- Structure: src/core/, src/models/, src/training/, src/inference/, experiments/, api/, config/, scripts/, tests/
## Errors Encountered
- None yet

## Status
**Currently in Phase 2** - Designing new folder structure</content>
<parameter name="filePath">c:\Users\tsuma.thomas\Documents\CoreOutline\transformer\task_plan.md
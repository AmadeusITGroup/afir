### Community guidelines
 * Be respectful to others
 * Be appreciative and welcoming
 * Don't be judgmental
 * Be patient and supportive to newcomers
 * Value each contribution, even if it's not perfect - we can work as a team to benefit even from a failed attempt as we can learn from it!
 * Look after one another - we are community of like-minded people who care about others, not only about ticking the boxes
 * A challenge is not a bad thing, as it leads to expanding the horizons, being to competitive leads to unhealthy situations - be reasonable here

### Code conventions
Please follow the standard python rules if possible:
  * existing conventions
  * [PEP8](https://www.python.org/dev/peps/pep-0008/)
  * [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

There is always room for opportunistic refactoring, but be careful and ensure the cosmetic changes have no adverse impact on performance or readability.

### Testing conventions
The project has a suite of unit tests. All existing tests must pass before submitting a PR. Any new feature, changed flow, or bug fix must be accompanied by new or updated tests.

Run the suite with:

```bash
pip install -r requirements-dev.txt
pytest
```

Please follow [pytest good practices](https://docs.pytest.org/en/stable/goodpractices.html).

**Formatting posture.** There is no `setup.cfg`, `pyproject.toml`, or `.flake8` in this repository. Running `flake8` with defaults applies a 79-column limit against black's 88, which produces roughly 13,600 pre-existing findings across the codebase, and `black src tests` would reformat files unrelated to a change. Format only the files you touched, and verify that a touched file introduces no new non-E501 flake8 findings. Repository-wide reformatting is a separate, coordinated commit and is not expected from individual contributors.

Make sure all new resources are checked in before submitting the PR.

**Measurement citations, and the one class of file that is deliberately absent.** Comments in `src/` and prose in `docs/architecture/` regularly attribute a number to a one-off script — `scripts/probe_*.py`, `scripts/measure_*.py`, or a run log under `scripts/out/`. Most of those are **not in the repository**, and that is the intent rather than an omission: they are single-use instruments pointed at a live backend, carrying one deployment's coordinates, and their output directory is gitignored (the `scripts/out/` entry in `.gitignore`). The artifact that ships is the measured **result**, recorded beside the value it justifies — so a comment reading *"measured in `scripts/probe_volume_semantics.py`: 768 KB accepted, 1024 KB answered HTTP 400"* is complete without the file, and the filename is an attribution rather than a path you can follow.

Two obligations follow, and they are the whole convention. **Put the number in the sentence that cites it** — a bare *"see `scripts/probe_x.py`"* says nothing in a fresh clone, and the fix is to move the measurement into the prose, not to commit the probe. **A script becomes tracked only when a document presents it as an instrument to RE-RUN**, not as provenance for a result already stated — and a tracked script must hold no deployment-specific literal, which is most of why so few of them qualify. So the rule above applies in full to everything a reviewer or a fresh clone needs in order to build, run and test — and stops at your own measurement scratch.

### Branching conventions
We are working with a single default branch (`main`) and one development branch to keep it simple. Open your PR against `main`.

### Commit-message conventions
 * Prefix each commit with the GitHub Issue ticket number if possible i.e. [ABC-123] New package nnn added to allow running bbb
 * Provide a high level summary of the changes. Try to be concise.
 * Make the commit messages meaningful. Don't skip them. They may be helpful during the code review and act as a passive documentation going forwards.

### Steps for creating good pull requests
  * State your intent is very clearly
  * If there is a need to provide a thorough explanation or refer to external sources, please do it - it will help in the review
  * Use the GitHub Issue ticket as a prefix in the PR title to ensure we get nice cross-references
  * If you are unsure about certain aspects, don't be scared of asking on the available forums ahead of creating the PR.
  * If you are aware of some drawback of the changes introduced be transparent about it, the reviewers will weigh pros and cons and your contribution may still be accepted
  * To stick to a reasonable number of commits, you may want to squash your PR using [the git  history rewriting technique](https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History)
  * You can use the PR as a medium for the conversation between yourself and the project maintainers. You can prefix the PR with a meaningful tag eg. [IDEA], [SUGGESTION], [REMARK] etc. In such case your PR may never be integrated if what you are proposing is not in line with the general direction the project is going to. However, it would be still a valuable resource to track the discussion that took place and it may save time for somebody who is heading in a similar direction.

### Expected timelines(SLAs) for the code review and the integration
 * The PRs should be reviewed within 1 week at least
 * The integration happens immediately after the PR is approved and merged into the target branch 

### How to submit feature requests
 * You may want to discuss the feature on the teams channel or other forums available for the project
 * Use GitHub Issues associated to this project
 * Link the PR(when the contribution is planned) with the GitHub issue if possible - it may give us more context and will make the case for the change stronger
 * Please be thorough with the description
 * Highlight a reasonable timescale you wish the feature to be integrated within - it is helpful when prioritizing 

### How to submit bug reports
 * Check carefully the documentation and by asking on the available forums if the behaviour you are experiencing is expected or if it is a bug
 * Use Issues board associated to the project
 * Link the PR(when the contribution is planned) with the bug report request if possible - it may give us more context
 * Please be thorough with the description
 * Highlight a reasonable timescale you wish the feature to be integrated within - it is helpful when prioritizing 

### How to submit security issue reports
* Engage with project maintainers. Do not publicly disclose anything before the patch is delivered.

### How to write documentation
  * README.md is where we describe the overview, the usage and the development practices
  * Use [markdown syntax](https://www.markdownguide.org/basic-syntax/) which is widely supported in the GitHub, BitBucket WebUI as well as in many IDEs
  * Use plain English and check your spelling prior to committing the change
  * Remember that good documentation is essential, so take time to do it properly

### Dependencies
Runtime dependencies are listed in [requirements.txt](./requirements.txt). Development, testing, and formatting dependencies are in [requirements-dev.txt](./requirements-dev.txt), which installs the runtime set as well (`pip install -r requirements-dev.txt` is the developer install command).

### Build process schedule
A Python wheel is produced when a PR is merged into the release branch.

### Contribution schedule
Contributions are reviewed and integrated asynchronously. There is no fixed sprint cadence. PRs are reviewed within approximately one week of submission.

### Road map
Feature priorities are discussed in GitHub Issues and evolve as the project develops. Contributions that align with open issues are particularly welcome.

### When the repositories will be closed to contributions
At this stage the repositories never get frozen.

### Time reporting
There is no budget behind this project. Potential contributors need to negotiate it with their line or project managers.

### Helpful links, information, and documentation
  * [Markdown syntax](https://www.markdownguide.org/basic-syntax/)
  * [PEP8](https://www.python.org/dev/peps/pep-0008/)
  * [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)



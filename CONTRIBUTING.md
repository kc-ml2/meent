# Contributing to meent

meent is a working scientific library, but it is still being stabilized toward `1.0.0`. Our development process therefore distinguishes between two statements:

- **A change has been implemented and reviewed.**
- **The integrated library has been scientifically validated and released.**

The purpose of this distinction is confidence and traceability, not additional bureaucracy. Most researchers only need to learn the contributor path: start from `dev`, make one focused change on a short-lived branch, present the evidence, and open a pull request to `dev`.

## How work reaches a release

```text
feature/*  -- PR + tests + review -->  dev  -- scientific validation -->  main  -- tag/release -->  PyPI
work in progress                    integrated next version              endorsed code             distributed package
```

There are two different gates:

- **Feature gate — `feature/* → dev`:** reviewers decide whether one change is clear, credible, and suitable for integration. Passing this gate does not yet make the change part of a release.
- **Release gate — `dev → main`:** the validation committee evaluates one exact integrated candidate. Passing this gate means that the group is prepared to release and endorse that code.

### Four rules to remember

1. **Start normal development from `dev`, not `main`.**
2. **Open feature and fix pull requests against `dev`.** Check the PR's GitHub **base** branch.
3. **Validate an exact commit, not the moving branch name `dev`.** A code change creates a new candidate.
4. **The validated commit, `main`, the Git tag, the GitHub release, and the PyPI package must describe the same code.**

Where practical, GitHub should enforce pull requests, CI checks, and required approvals. Contributors should not have to memorize safeguards that the repository can apply automatically.

## What the branches mean

| Branch | Meaning | Working rule | Normal destination |
| --- | --- | --- | --- |
| `feature/*`, `fix/*`, `docs/*` | One focused change under active work | The branch owner may push directly while developing | PR to `dev` |
| `dev` | Integrated work intended for a future release | No direct pushes; changes enter through reviewed PRs and required checks | Release PR to `main` |
| `main` | Scientifically validated and released code | No direct pushes; release approval and required checks apply | Tag and distribute |
| `hotfix/*` | A narrow correction to the currently released version | Branch from `main` and validate the specific correction | PR to `main`, then propagate to `dev` |
| `release/X.Y` *(optional)* | A frozen candidate while `dev` continues moving | Use only when freezing `dev` becomes disruptive | PR to `main`; fixes must also reach `dev` |

### About meent's pre-release status

The GitHub **Pre-release** label describes meent's current maturity and intended audience. We use GitHub and PyPI to collaborate with a known research community, while the public API and supported behavior may still change before `1.0.0`.

A version such as `0.14.0` is nevertheless a real, immutable research distribution. Record the package version in experiment notes; when using code directly from `dev`, also record the commit SHA. The Pre-release label is not access control: the public repository and PyPI package remain visible to others.

`1.0.0` will represent a broader stability and compatibility promise. A version such as `1.0.0rc1` should be used only when the intended 1.0 features and public interface are effectively frozen.

## Making one contribution

> **Your responsibility:** implement one coherent change, explain its scientific or user-visible effect, and provide the best evidence available to you. Maintainers take responsibility for release mechanics after the change enters `dev`.

The normal path is:

1. Update your local `dev` branch.
2. Create a short-lived branch, for example `feature/anisotropic-numpy`, `fix/jax-device-detection`, or `docs/material-table-format`.
3. Make the change and run the relevant checks you can run.
4. Push the branch and open a pull request with **base: `dev`**.
5. Explain what changed, which outputs or interfaces may change, and how you checked the result.
6. Respond to review. A maintainer merges an accepted PR into `dev`.

You may use the Git command line, GitHub Desktop, or an IDE. Fluency with Git commands is not a prerequisite for contributing scientific knowledge.

### Before you open the pull request

- [ ] The change has one clear purpose; unrelated cleanup is separated when practical.
- [ ] The pull request targets `dev`.
- [ ] The description identifies possible changes to numerical results, conventions, shapes, units, gradients, or public APIs.
- [ ] Relevant checks were run, and the results are summarized rather than described only as “tested.”
- [ ] Known limitations, unsupported backends, and untested conditions are stated honestly.

Draft pull requests are welcome when the formulation, convention, implementation direction, or validation method still needs discussion.

## Reviewing a contribution

Approving a feature pull request means:

> **This change is suitable to integrate into `dev` with the evidence currently shown.**

It is not yet an endorsement of a released version. Review the change through three questions:

1. **Intent:** Is the intended scientific or user-facing behavior clear? Is a deliberate convention change distinguished from a bug fix?
2. **Implementation:** Is the change plausible, focused, maintainable, and consistent with the affected backends and public interface?
3. **Evidence:** Do the checks observe the quantities that could be wrong, using appropriate cases and defensible tolerances?

End the review with a clear outcome: **approve for `dev`**, **request specific changes**, or **request additional evidence or committee discussion**.

## Preparing a validated release

Scientific validation applies to a specific commit SHA. The name `dev` alone is insufficient because the branch can move while validation is in progress.

To prepare a release:

1. Open a release pull request from `dev` to `main` and record the candidate commit SHA.
2. Initially, freeze `dev` while validation is active. Do not merge unrelated work into the candidate.
3. Assign validation areas and owners in the release pull request.
4. Run the required automated suite and the agreed scientific checks.
5. Record the environment, cases, tolerances, results, exceptions, and unresolved risks.
6. Proceed only after the required committee members approve that exact candidate.

A fix or cleanup made after validation begins produces a new commit and therefore a new candidate. Re-run the affected checks and renew the relevant approvals.

## Merge conventions

- **Feature or fix PR → `dev`:** use squash merge by default, so one coherent change remains easy to identify and revert. Preserve multiple commits when their history contains meaningful experiments or review decisions.
- **Release PR `dev → main`:** use a normal merge commit rather than squash. This preserves branch ancestry and keeps already released feature commits from appearing as new in the next release comparison.
- **Hotfix PR → `main`:** merge the narrow correction, publish a patch release, and then propagate the identical fix to `dev`.

## Responsibilities at a glance

| Role | Main responsibility |
| --- | --- |
| Contributor | Start from `dev`, make one focused change, explain its impact and evidence, and open the PR to `dev`. |
| Reviewer | Evaluate intent, implementation, and evidence. Approval means suitable for integration into `dev`. |
| Validation committee | Evaluate one exact integrated candidate before it enters `main`. |
| Maintainer | Protect the branch process, record approval, merge, tag, and distribute the same validated code. |

**Automated checks ask whether the software behaved as the tests specify. Scientific validation also asks whether we specified and observed the right scientific behavior.**

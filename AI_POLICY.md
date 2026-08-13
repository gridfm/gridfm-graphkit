# AI Policy

This policy describes how AI-assisted tools (code assistants, chatbots, agents,
and other generative models) may be used when contributing to
**gridfm-graphkit**. It complements the [Contribution Guidelines](CONTRIBUTING.md),
the [Code of Conduct](CODE_OF_CONDUCT.md), and the project [Governance](GOVERNANCE.md).

The goal is simple: AI tools are welcome as an aid, but every contributor remains
fully accountable for what they submit.

## Guiding principles

* **You own your contribution.** Regardless of the tools used to produce it, you
  are responsible for the correctness, licensing, and quality of any code,
  documentation, or other artifact you submit. "The AI wrote it" is never an
  explanation for a defect or a license violation.
* **Understand before you submit.** Do not open a pull request containing code you
  cannot explain or would not be able to maintain and debug yourself.
* **Human review is required.** AI output must be reviewed by a human before
  submission, and all contributions continue to go through the normal committer
  review process.

## Permitted uses

AI tools may be used to:

* Draft, refactor, or explain code and tests.
* Generate or improve documentation, docstrings, and comments.
* Suggest fixes, investigate bugs, or explore design alternatives.
* Assist with reviews, triage, and issue summarization.

## Requirements for AI-assisted contributions

* **Licensing and provenance.** Do not submit AI-generated content that reproduces
  substantial portions of code whose license is incompatible with this project's
  [LICENSE](LICENSE). Contributions must be yours to give under the project's
  license and the [DCO](CONTRIBUTING.md) sign-off you provide.
* **Correctness and testing.** AI-assisted code is held to the same standard as any
  other contribution: it must build, pass CI, and be covered by appropriate tests.
* **No unverified claims.** Do not paste AI-generated benchmarks, citations, or
  factual claims into issues, PRs, or docs without verifying them.
* **Security.** Do not paste secrets, credentials, private data, or unpublished
  security-sensitive information into third-party AI services. Review AI-suggested
  code for introduced vulnerabilities before submitting.

## Disclosure

Disclosure of AI assistance is encouraged but not mandatory. If AI tools played a
significant role in generating a contribution, a brief note in the pull request
description helps reviewers focus their attention. What matters most is that the
contribution meets the project's quality bar, not which tools produced it.

## Maintainers and automated agents

Committers and maintainers may use AI tools to assist with reviews, triage, and
routine maintenance. Automated agents (bots) acting on the repository must be
transparent about their nature and remain subject to human oversight; a human is
always accountable for merges and releases.

## Questions

If you are unsure whether a particular use of AI tooling is appropriate, ask on the
project's community channels or open a discussion before submitting. See
[SUPPORT.md](SUPPORT.md) for how to reach the maintainers.

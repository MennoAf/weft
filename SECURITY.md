# Security Policy

## Supported versions

Security reports are accepted for the `1.0.x` series.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting for this repository: open the **Security** tab and choose **Report a vulnerability**. This sends the report privately to the repository administrators. No security-reporting email address is published.

Please include, when safe to do so:

- The affected Weft version and commit, if known.
- The impact and conditions required to reproduce the issue.
- A minimal reproduction or proof of concept that does not access another person's data.

Before submitting, remove memory contents, credentials, tokens, personal data, and other sensitive information from logs, examples, and reproduction steps. Do not include real user data in a report.

Reports are reviewed as maintainer capacity allows. Maintainers will acknowledge reports and coordinate with the reporter about investigation and any appropriate next steps. No fixed response or resolution time is promised.

## Data and deployment notes

Weft stores persistent agent memory. Operators are responsible for protecting the database and its backups, configuring transport and encryption appropriately for their deployment, and safeguarding provider credentials. If you suspect memory data, credentials, or another deployment resource has been exposed, report it through the same private GitHub channel described above.

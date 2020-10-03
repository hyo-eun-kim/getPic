# GetPic
Tobigs 10th Conference

(배너 png파일 주소 들어갈자리)


## Table of Contents
- [1. Short Description](#1-short-description)
  - [Example](#example)
- [2. ChatBot](#2-chatbot)
  - [2.1. Install](#21-install)
  - [2.2. Test](#22-test)
- [3. Web](#3-web)
  - [3.1. Getting Started in Web](#31-getting-started-in-web)
  - [3.2. Install](#32-install)
  - [3.3. Test](#33-test)
- [4. API](#4-api)
- [5. Contributing](#5-contributing)
  


## 1. Short Description
**Status:** Required.

**Requirements:**
- Must not have its own title.
- Must be less than 120 characters.
- Must not start with `> `
- Must be on its own line.
- Must match the description in the packager manager's `description` field.
- Must match GitHub's description (if on GitHub).

**Suggestions:**
- Use [gh-description](https://github.com/RichardLitt/gh-description) to set and get GitHub description.
- Use `npm show . description` to show the description from a local [npm](https://npmjs.com) package.

### Example


  
## 2. ChatBot

### 2.1. Install

### 2.2. Test

**Status:** Required by default, optional for [documentation repositories](#definitions).

**Requirements:**
- Code block illustrating how to install.

**Subsections:**
- `Dependencies`. Required if there are unusual dependencies or dependencies that must be manually installed.

**Suggestions:**
- Link to prerequisite sites for programming language: [npmjs](https://npmjs.com), [godocs](https://godoc.org), etc.
- Include any system-specific information needed for installation.
- An `Updating` section would be useful for most packages, if there are multiple versions which the user may interface with.




## 3. Web

### 3.1. Getting Started in Web

### 3.2. Install

### 3.3. Test

**Status:** Optional.

**Requirements:**
- Describe exported functions and objects.

**Suggestions:**
- Describe signatures, return types, callbacks, and events.
- Cover types covered where not obvious.
- Describe caveats.
- If using an external API generator (like go-doc, js-doc, or so on), point to an external `API.md` file. This can be the only item in the section, if present.

## 4. API
**Status:** Optional.

**Requirements:**
- Describe exported functions and objects.

**Suggestions:**
- Describe signatures, return types, callbacks, and events.
- Cover types covered where not obvious.
- Describe caveats.
- If using an external API generator (like go-doc, js-doc, or so on), point to an external `API.md` file. This can be the only item in the section, if present.


## 5. Contributing
**Status**: Required.

**Requirements:**
- State where users can ask questions.
- State whether PRs are accepted.
- List any requirements for contributing; for instance, having a sign-off on commits.

**Suggestions:**
- Link to a CONTRIBUTING file -- if there is one.
- Be as friendly as possible.
- Link to the GitHub issues.
- Link to a Code of Conduct. A CoC is often in the Contributing section or document, or set elsewhere for an entire organization, so it may not be necessary to include the entire file in each repository. However, it is highly recommended to always link to the code, wherever it lives.
- A subsection for listing contributors is also welcome here.


# Yong-Hwan Lee · AI Engineer Portfolio

[Magic UI Portfolio](https://github.com/magicuidesign/portfolio)의 간결한 레이아웃을 바탕으로, AI 프로젝트와 기술 기록을 소개하는 포트폴리오입니다. Next.js 16으로 정적 사이트를 생성해 GitHub Pages에 배포합니다.

## 로컬 실행

Node.js 24와 npm을 설치한 뒤 저장소 폴더에서 실행합니다. Windows PowerShell에서는 다음 명령을 사용합니다.

```powershell
npm.cmd ci
npm.cmd run dev
```

브라우저에서 [http://localhost:3000](http://localhost:3000)을 엽니다. macOS/Linux에서는 `npm.cmd` 대신 `npm`을 사용합니다.

## 내용 수정

| 수정할 내용 | 파일 |
| --- | --- |
| 소개, 학력, 기술, 연락처 | `src/data/profile.ts` |
| 프로젝트 카드의 요약과 표시 정보 | `src/data/projects.ts` |
| 프로젝트 상세 본문과 첨부 이미지·PDF | `content/projects/<slug>/index.md`와 같은 폴더의 첨부 파일 |
| 기술 기록 본문 | `content/studies/*.md` |

기존 Markdown을 계속 편집하면 됩니다. `draft: true`인 글은 공개 사이트에서 제외됩니다. 기존 프로젝트와 기술 기록의 주소를 유지하려면 폴더명·파일명을 변경하지 마세요.

`public/`은 실행·빌드 시 생성되는 에셋 폴더입니다. 직접 수정하지 말고 원본 콘텐츠와 에셋을 수정하세요.

### Project ordering

`featuredSlugs` in `src/data/projects.ts` controls the homepage's selected work
and the leading order of the project index and “Keep exploring” navigation.
Unfeatured projects follow in descending publication-date order. Studies remain
chronological. Use real publication dates in Markdown; change `featuredSlugs`
to curate the project order.

## 배포 전 확인

```powershell
npm.cmd run lint
npm.cmd run typecheck
npm.cmd run build
npm.cmd run verify
```

`build`는 정적 파일을 `out/`에 생성하고, `verify`는 생성된 페이지와 내부 링크를 확인합니다.

빌드 결과를 로컬에서 확인하려면 `npm.cmd run preview`를 실행하고 [http://127.0.0.1:3000](http://127.0.0.1:3000)을 엽니다. 개발 서버와 같은 포트를 사용하므로 두 서버를 동시에 실행하지 마세요.

## GitHub Pages 배포

[저장소 Pages 설정](https://github.com/kapshaul/kapshaul.github.io/settings/pages)에서 게시 소스를 **GitHub Actions**로 지정합니다. 설정은 워크플로가 자동으로 변경하지 않습니다.

- `main` 대상 Pull Request: lint, 타입 검사, 빌드, 정적 링크 검증을 실행합니다.
- `main`에 push 또는 `main`에서 수동 실행: 검증을 통과한 `out/`을 GitHub Pages에 배포합니다.
- 다른 브랜치에서 수동 실행: 검증만 수행합니다.

워크플로는 기존 경로인 `.github/workflows/hugo.yml`에 있습니다. 파일명과 달리 Hugo는 실행하지 않습니다. 수동 실행(`workflow_dispatch`)은 기본 브랜치에 같은 경로의 워크플로가 있어야 하므로, 이 변경이 `main`에 반영되기 전까지는 경로를 바꾸지 마세요. 마이그레이션 범위와 배포 확인 사항은 [MIGRATION.md](MIGRATION.md)를 참고하세요.

## 출처와 라이선스

디자인·컴포넌트 출처는 [Magic UI Portfolio](https://github.com/magicuidesign/portfolio)입니다. 관련 라이선스는 `licenses/`에 보존합니다. 현재 사이트는 Next.js로 빌드합니다. 이전 Hugo 구성은 삭제했고, 원본 콘텐츠는 `content/`와 `static/`에 남아 있습니다.

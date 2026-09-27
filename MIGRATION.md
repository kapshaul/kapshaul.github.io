# Magic UI 포트폴리오 마이그레이션

## 변경 범위

Hugo/PaperMod 화면을 Magic UI Portfolio를 바탕으로 한 Next.js 정적 사이트로 전환합니다. 메인 화면에서 소개와 대표 프로젝트를 훑어보고, 프로젝트 상세에서 기존 기술 설명·이미지·PDF를 읽을 수 있는 구조입니다.

- 프로필과 카드 표시 정보는 `src/data/`에서 관리합니다.
- 프로젝트·기술 기록 본문은 기존 `content/projects/`, `content/studies/`의 Markdown을 사용합니다.
- 기존 공개 프로젝트·기술 기록 URL을 유지합니다. `draft: true`인 문서는 배포하지 않습니다.
- 기존 템플릿의 예시 콘텐츠(논문·책·강의·데이터, 아카이브·위치 페이지, 예시 이력서 `cv.pdf`, 예시 논문 `jmp.pdf`)와 본문에서 참조하지 않는 이미지·PDF는 삭제했습니다.
- 새 빌드에 쓰이지 않는 Hugo 전용 파일(`config.yml`, `themes/`, `layouts/`, `assets/`, `archetypes/`, `resources/`, `hugo.exe`)도 삭제했습니다. 게시되는 원본 콘텐츠는 `content/`와 `static/`에 있습니다.

## 배포 방식

GitHub Actions가 Node.js 24에서 의존성 설치, lint, 타입 검사, 정적 빌드와 링크 검증을 실행합니다. Pull Request에서는 배포하지 않으며, `main`의 push 또는 `main`에서 수동 실행할 때 검증된 `out/`을 배포합니다.

GitHub Pages의 게시 소스는 **GitHub Actions**여야 합니다. 이 변경은 Pages를 자동으로 활성화하지 않습니다. 워크플로 파일을 로컬에서 수정하는 것만으로 현재 공개 사이트가 바뀌지는 않습니다.

## 공개 전 확인

1. `npm.cmd ci` 후 `npm.cmd run dev`로 소개, 연락처와 프로젝트 설명을 확인합니다.
2. 대표 프로젝트 상세에서 이미지, 수식, 코드와 PDF 링크를 확인합니다.
3. 모바일 화면과 밝은/어두운 테마를 확인합니다.
4. `npm.cmd run lint`, `npm.cmd run typecheck`, `npm.cmd run build`, `npm.cmd run verify`를 실행합니다.
5. 게시할 준비가 되면 `main`에 반영하고, Actions의 배포 성공 여부와 공개 사이트를 확인합니다.

실제 이력서를 제공하려면 본인의 최신 PDF를 추가하고 해당 링크를 연결하세요. 프로필에 없는 경력이나 검증되지 않은 성과 수치를 템플릿 예시에서 가져오지 않습니다.

## 이전 버전으로 돌아가기

Hugo 구성은 이 저장소에서 삭제했습니다. 이전 방식으로 배포하려면 원격 `main`의 마이그레이션 이전 커밋(`8425831`)에서 Hugo 소스와 워크플로를 복원한 뒤 `main`에 반영합니다. 현재 배포 워크플로는 Next.js 전용이므로, Hugo 파일만 복원해서는 Hugo 사이트로 배포되지 않습니다.

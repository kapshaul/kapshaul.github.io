import type { ProjectDisplay } from "@/data/projects";

export default function ProjectVideo({
  video,
}: {
  video: NonNullable<ProjectDisplay["video"]>;
}) {
  return (
    <video
      controls
      muted
      playsInline
      preload="metadata"
      poster={video.poster}
      width={video.width}
      height={video.height}
      aria-label={video.title}
      className="block h-auto w-full"
    >
      <source src={video.src} type="video/mp4" />
      <a href={video.src}>Download the JARVIS demo video</a>.
    </video>
  );
}

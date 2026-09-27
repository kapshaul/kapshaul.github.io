import Link from "next/link";
export default function NotFound() {
  return (
    <main id="main" className="py-16 text-center">
      <p className="font-mono text-sm text-muted-foreground">404</p>
      <h1 className="mt-4 text-3xl font-semibold tracking-tight">
        This page isn’t here.
      </h1>
      <p className="mt-4 text-sm text-muted-foreground">
        You can find my work and technical notes below.
      </p>
      <div className="mt-7 flex justify-center gap-5 text-sm">
        <Link className="underline underline-offset-4" href="/">
          Home
        </Link>
        <Link className="underline underline-offset-4" href="/projects/">
          Projects
        </Link>
        <Link className="underline underline-offset-4" href="/studies/">
          Studies
        </Link>
      </div>
    </main>
  );
}

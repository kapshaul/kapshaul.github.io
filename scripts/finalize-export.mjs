import fs from "node:fs";
import path from "node:path";

// Next also copies the legacy root static/ directory. Hugo's originals remain
// in source control; the synchronized public/ tree is the deployed asset source.
const root = fs.realpathSync(process.cwd());
const output = path.join(root, "out");
const duplicate = path.join(output, "static");
if (!fs.existsSync(path.join(output, "index.html")))
  throw new Error("Build the static site first");
if (fs.realpathSync(output) !== output || path.dirname(duplicate) !== output)
  throw new Error("Unsafe export cleanup path");
if (fs.existsSync(duplicate)) {
  if (
    fs.lstatSync(duplicate).isSymbolicLink() ||
    fs.realpathSync(duplicate) !== duplicate
  )
    throw new Error("Refusing to clean a linked output directory");
  fs.rmSync(duplicate, { recursive: true });
}
console.log("Static export finalized; duplicate static/ copy removed.");

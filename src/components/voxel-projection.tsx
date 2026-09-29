"use client";

import { Fragment, useId, useState } from "react";
import styles from "./voxel-projection.module.css";

type Direction = "horizontal" | "vertical";
type Point = [number, number];

// Screen map for one voxel layer: u runs along columns, v along rows, z up.
// A 45-degree camera elevation keeps a mid-height ray clear of cube edges,
// seams and top-face labels (true isometric puts it exactly on an edge).
const W = 56;
const H = 40;
const Z = 56;
const ORIGIN_X = 270;
const ORIGIN_Y = 72;
const SIZE = 0.88; // cube edge in cell units; the remainder is the seam
const GAP = (1 - SIZE) / 2;
const MID = SIZE / 2; // ray height
// A mid-height point projects onto the top plane shifted by LIFT along u and v.
const LIFT = (MID * Z) / (2 * H);
const RAY_START = -1;
const RAY_TIP = 3.95;
const AXIS_LABEL_AT = -1.2;

// Row indices of the fixed 16 × 9 sensing matrix A for each axis-aligned path.
const OBSERVATION: Record<Direction, readonly number[]> = {
  horizontal: [3, 2, 1],
  vertical: [9, 10, 11],
};

function project(u: number, v: number, z: number): Point {
  return [ORIGIN_X + (u - v) * W, ORIGIN_Y + (u + v) * H - z * Z];
}

function round(value: number): number {
  return Math.round(value * 10) / 10;
}

function polygon(list: Point[]): string {
  return list.map(([x, y]) => `${round(x)},${round(y)}`).join(" ");
}

function plate(from: number, to: number): string {
  return polygon([
    project(from, from, 0),
    project(to, from, 0),
    project(to, to, 0),
    project(from, to, 0),
  ]);
}

const FLOOR = plate(-0.32, 3.32);

// Orientation triad in the empty lower-left corner, clear of every Y chip:
// x along columns, y along rows, z up (same screen directions as project()).
const TRIAD_ORIGIN: Point = [48, 312];
type TriadAxis = {
  name: string;
  vector: Point;
  label: Point;
  anchor: "start" | "middle" | "end";
};
const TRIAD_AXES: TriadAxis[] = [
  { name: "x", vector: [24, 17], label: [80, 335], anchor: "middle" },
  { name: "y", vector: [-24, 17], label: [16, 335], anchor: "middle" },
  { name: "z", vector: [0, -30], label: [55, 287], anchor: "start" },
];
const TRIAD = TRIAD_AXES.map(({ name, vector, label, anchor }) => {
  const [dx, dy] = vector;
  const length = Math.hypot(dx, dy);
  const [ux, uy] = [dx / length, dy / length];
  const tip: Point = [TRIAD_ORIGIN[0] + dx, TRIAD_ORIGIN[1] + dy];
  const at = (distance: number, side: number): Point => [
    round(tip[0] - ux * distance - uy * side),
    round(tip[1] - uy * distance + ux * side),
  ];
  return {
    name,
    label,
    anchor,
    shaftEnd: at(5, 0),
    head: polygon([tip, at(7, 2.6), at(7, -2.6)]),
  };
});

// Row-major voxel numbering, drawn back to front so nearer cubes overlap.
const CUBES = Array.from({ length: 9 }, (_, k) => {
  const row = Math.floor(k / 3);
  const col = k % 3;
  const [u0, u1, v0, v1] = [col + GAP, col + 1 - GAP, row + GAP, row + 1 - GAP];
  const [labelX, labelY] = project(col + 0.5, row + 0.5, SIZE);
  return {
    row,
    col,
    voxel: k + 1,
    top: polygon([
      project(u0, v0, SIZE),
      project(u1, v0, SIZE),
      project(u1, v1, SIZE),
      project(u0, v1, SIZE),
    ]),
    right: polygon([
      project(u1, v0, SIZE),
      project(u1, v1, SIZE),
      project(u1, v1, 0),
      project(u1, v0, 0),
    ]),
    left: polygon([
      project(u0, v1, SIZE),
      project(u1, v1, SIZE),
      project(u1, v1, 0),
      project(u0, v1, 0),
    ]),
    label: [round(labelX), round(labelY - 4)] as Point,
  };
}).sort((a, b) => a.row + a.col - (b.row + b.col));

function rayPoint(direction: Direction, index: number, t: number): Point {
  const [x, y] =
    direction === "horizontal"
      ? project(t, index + 0.5, MID)
      : project(index + 0.5, t, MID);
  return [round(x), round(y)];
}

function crossedVoxels(direction: Direction, index: number): number[] {
  return [0, 1, 2].map((step) =>
    direction === "horizontal" ? index * 3 + step + 1 : step * 3 + index + 1,
  );
}

function pathName(direction: Direction, index: number): string {
  return `${direction === "horizontal" ? "Row" : "Column"} ${index + 1}`;
}

function Segment({
  from,
  to,
  className,
}: {
  from: Point;
  to: Point;
  className: string;
}) {
  return (
    <line
      className={className}
      x1={from[0]}
      y1={from[1]}
      x2={to[0]}
      y2={to[1]}
    />
  );
}

function PathIcon({ direction }: { direction: Direction }) {
  return (
    <svg
      className={styles.icon}
      viewBox="0 0 14 14"
      width="14"
      height="14"
      aria-hidden="true"
      focusable="false"
    >
      {[0, 1, 2].flatMap((row) =>
        [0, 1, 2].map((col) => (
          <rect
            key={`${row}-${col}`}
            x={col * 4.6 + 0.5}
            y={row * 4.6 + 0.5}
            width="3.8"
            height="3.8"
            rx="0.8"
          />
        )),
      )}
      {direction === "horizontal" ? (
        <line x1="0" y1="7" x2="14" y2="7" />
      ) : (
        <line x1="7" y1="0" x2="7" y2="14" />
      )}
    </svg>
  );
}

export function VoxelProjection() {
  const id = useId();
  const titleId = `${id}-title`;
  const descId = `${id}-desc`;
  const [direction, setDirection] = useState<Direction>("horizontal");
  const [index, setIndex] = useState(1);

  const observation = OBSERVATION[direction][index];
  const voxels = crossedVoxels(direction, index);
  const path = pathName(direction, index);
  const isSelected = (row: number, col: number) =>
    (direction === "horizontal" ? row : col) === index;

  // Incoming ray is drawn behind the cubes, so they occlude it naturally.
  // The dashed run marks the hidden part from where it disappears to the exit.
  const start = rayPoint(direction, index, RAY_START);
  const entry = rayPoint(direction, index, GAP);
  const hidden = rayPoint(direction, index, GAP - LIFT);
  const exit = rayPoint(direction, index, 3 - GAP);
  const tip = rayPoint(direction, index, RAY_TIP);
  const back = rayPoint(direction, index, RAY_TIP - 1);
  const length = Math.hypot(tip[0] - back[0], tip[1] - back[1]);
  const unit: Point = [
    (tip[0] - back[0]) / length,
    (tip[1] - back[1]) / length,
  ];
  const along = (from: Point, distance: number, side = 0): Point => [
    round(from[0] + unit[0] * distance - unit[1] * side),
    round(from[1] + unit[1] * distance + unit[0] * side),
  ];
  const shaftEnd = along(tip, -10);
  const arrowHead = polygon([tip, along(tip, -12, 5), along(tip, -12, -5)]);
  const chip = along(tip, 34);
  const chipWidth = 40 + 11 * String(observation).length;

  const voxelList = voxels.map((j) => `p${j}`);
  const spokenVoxels = `${voxelList.slice(0, 2).join(", ")} and ${voxelList[2]}`;
  const expectedLabel = `E of Y ${observation} equals ${voxels
    .map((j) => `a ${observation},${j} times x ${j}`)
    .join(" plus ")}`;
  const observedLabel = `Y ${observation} equals ${voxels
    .map((j) => `N ${observation},${j}`)
    .join(" plus ")}`;

  return (
    <figure className={styles.root}>
      <div className={styles.header}>
        <span className={styles.eyebrow}>Interactive · one 3 × 3 layer</span>
        <span className={styles.title}>Projection through a voxel layer</span>
      </div>

      <div className={styles.controls}>
        <div
          className={styles.segmented}
          role="group"
          aria-label="Path direction"
        >
          {(["horizontal", "vertical"] as const).map((value) => (
            <button
              key={value}
              type="button"
              className={styles.segment}
              aria-pressed={direction === value}
              onClick={() => setDirection(value)}
            >
              <PathIcon direction={value} />
              {value === "horizontal" ? "Horizontal" : "Vertical"}
            </button>
          ))}
        </div>
        <div
          className={styles.segmented}
          role="group"
          aria-label={direction === "horizontal" ? "Row" : "Column"}
        >
          {[0, 1, 2].map((value) => (
            <button
              key={value}
              type="button"
              className={styles.segment}
              aria-pressed={index === value}
              onClick={() => setIndex(value)}
            >
              {pathName(direction, value)}
            </button>
          ))}
        </div>
      </div>

      <div className={styles.stage}>
        <svg
          className={styles.svg}
          viewBox="0 0 540 348"
          role="img"
          aria-labelledby={titleId}
          aria-describedby={descId}
        >
          <title id={titleId}>
            3 × 3 voxel layer drawn as cubes with one measurement path
          </title>
          <desc id={descId}>
            {`Nine cubes p1 to p9, numbered row by row, with unknown intensities x1 to x9. A ${direction} path through ${path.toLowerCase()} crosses ${spokenVoxels} at mid-height and ends at observation Y${observation}. The other six voxels are off the path. Axis triad: x runs along columns, y along rows, z up.`}
          </desc>

          <polygon className={styles.floor} points={FLOOR} />

          <g className={styles.triad} aria-hidden="true">
            {TRIAD.map((axis) => (
              <Fragment key={axis.name}>
                <Segment
                  from={TRIAD_ORIGIN}
                  to={axis.shaftEnd}
                  className={styles.triadLine}
                />
                <polygon className={styles.triadHead} points={axis.head} />
                <text
                  className={styles.triadLabel}
                  x={axis.label[0]}
                  y={axis.label[1]}
                  textAnchor={axis.anchor}
                  dominantBaseline="central"
                >
                  {axis.name}
                </text>
              </Fragment>
            ))}
            <circle
              className={styles.triadHead}
              cx={TRIAD_ORIGIN[0]}
              cy={TRIAD_ORIGIN[1]}
              r="1.8"
            />
          </g>

          <Segment from={start} to={entry} className={styles.rayHalo} />
          <Segment from={start} to={entry} className={styles.ray} />
          <circle className={styles.source} cx={start[0]} cy={start[1]} r="3.5" />

          {CUBES.map((cube) => {
            const selected = isSelected(cube.row, cube.col);
            return (
              <g
                key={cube.voxel}
                className={styles.cube}
                data-selected={selected || undefined}
              >
                <polygon className={styles.left} points={cube.left} />
                <polygon className={styles.right} points={cube.right} />
                <polygon className={styles.top} points={cube.top} />
                <text
                  className={styles.voxelLabel}
                  x={cube.label[0]}
                  y={cube.label[1]}
                  textAnchor="middle"
                  dominantBaseline="central"
                >
                  <tspan fontStyle="italic">p</tspan>
                  <tspan className={styles.sub} dy="0.3em">
                    {cube.voxel}
                  </tspan>
                </text>
              </g>
            );
          })}

          <Segment from={hidden} to={exit} className={styles.rayHidden} />
          <Segment from={exit} to={shaftEnd} className={styles.rayHalo} />
          <Segment from={exit} to={shaftEnd} className={styles.ray} />
          <circle className={styles.exit} cx={exit[0]} cy={exit[1]} r="2.5" />
          <polygon className={styles.arrow} points={arrowHead} />

          <rect
            className={styles.chip}
            x={round(chip[0] - chipWidth / 2)}
            y={round(chip[1] - 18)}
            width={chipWidth}
            height="36"
            rx="1.5"
          />
          <text
            className={styles.chipText}
            x={round(chip[0] - 2)}
            y={chip[1]}
            textAnchor="middle"
            dominantBaseline="central"
          >
            <tspan fontStyle="italic">Y</tspan>
            <tspan className={styles.sub} dy="0.3em">
              {observation}
            </tspan>
          </text>

          {[0, 1, 2].map((value) => {
            const [x, y] = rayPoint(direction, value, AXIS_LABEL_AT);
            return (
              <text
                key={value}
                className={styles.axisLabel}
                data-selected={value === index || undefined}
                x={direction === "horizontal" ? x - 4 : x + 4}
                y={y}
                textAnchor={direction === "horizontal" ? "end" : "start"}
                dominantBaseline="central"
              >
                {direction === "horizontal" ? "Row" : "Col"} {value + 1}
              </text>
            );
          })}
        </svg>
      </div>

      <div className={styles.readout}>
        <div className={styles.readoutHead}>
          <span className={styles.tag}>
            <i>Y</i>
            <sub>{observation}</sub>
          </span>
          <span>
            {direction === "horizontal" ? "Horizontal" : "Vertical"} path
            through {path.toLowerCase()} crosses{" "}
            {voxels.map((j, n) => (
              <Fragment key={j}>
                {n === 2 ? " and " : n === 1 ? ", " : ""}
                <span className={styles.math}>
                  <i>p</i>
                  <sub>{j}</sub>
                </span>
              </Fragment>
            ))}
            ; the other six{" "}
            <span className={styles.math}>
              <i>a</i>
              <sub>
                {observation},<i>j</i>
              </sub>
            </span>{" "}
            are zero.
          </span>
        </div>
        <dl className={styles.equations}>
          <div className={styles.equationRow}>
            <dt>Expected count</dt>
            <dd>
              <span
                className={styles.math}
                role="math"
                aria-label={expectedLabel}
              >
                <span className={styles.term} aria-hidden="true">
                  E[<i>Y</i>
                  <sub>{observation}</sub>] =
                </span>
                {voxels.map((j, n) => (
                  <Fragment key={j}>
                    {" "}
                    <span className={styles.term} aria-hidden="true">
                      {n > 0 && "+ "}
                      <i>a</i>
                      <sub>
                        {observation},{j}
                      </sub>
                      <i>x</i>
                      <sub>{j}</sub>
                    </span>
                  </Fragment>
                ))}
              </span>
            </dd>
          </div>
          <div className={styles.equationRow}>
            <dt>Observed count</dt>
            <dd>
              <span
                className={styles.math}
                role="math"
                aria-label={observedLabel}
              >
                <span className={styles.term} aria-hidden="true">
                  <i>Y</i>
                  <sub>{observation}</sub> =
                </span>
                {voxels.map((j, n) => (
                  <Fragment key={j}>
                    {" "}
                    <span className={styles.term} aria-hidden="true">
                      {n > 0 && "+ "}
                      <i>N</i>
                      <sub>
                        {observation},{j}
                      </sub>
                    </span>
                  </Fragment>
                ))}
              </span>
              <span className={styles.hint}>
                A random Poisson draw whose mean is E[<i>Y</i>
                <sub>{observation}</sub>].
              </span>
            </dd>
          </div>
        </dl>
        <p className={styles.srOnly} aria-live="polite" aria-atomic="true">
          {`${path} selected. Observation Y${observation} combines voxels ${spokenVoxels}. Expected count: ${expectedLabel}.`}
        </p>
      </div>

      <p className={styles.note}>
        Only the total <i>Y</i>
        <sub>
          <i>i</i>
        </sub>{" "}
        is observed, never the hidden per-voxel counts <i>N</i>
        <sub>
          <i>ij</i>
        </sub>
        . The E-step splits <i>Y</i>
        <sub>
          <i>i</i>
        </sub>{" "}
        among the crossed voxels in proportion to <i>a</i>
        <sub>
          <i>ij</i>
        </sub>
        <i>x</i>
        <sub>
          <i>j</i>
        </sub>{" "}
        at the current estimate; the M-step uses those estimated shares to
        update each intensity <i>x</i>
        <sub>
          <i>j</i>
        </sub>
        .
      </p>

      <figcaption className={styles.caption}>
        A single 3 × 3 voxel layer (<i>p</i>
        <sub>1</sub>–<i>p</i>
        <sub>9</sub>, numbered row by row) drawn as 3D cubes, with one
        axis-aligned path selected at a time. Each path is one selected row of
        the fixed binary sensing matrix <i>A</i>, not a scanner geometry.
        Dashed segments trace the path inside the layer.
      </figcaption>
    </figure>
  );
}

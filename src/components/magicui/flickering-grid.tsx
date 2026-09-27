"use client";

import { useEffect, useRef } from "react";

// Adapted from Magic UI's FlickeringGrid. See licenses/Magic-UI-Portfolio.txt.
export function FlickeringGrid() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) return;

    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    const squareSize = 2;
    const spacing = 4;
    const maxOpacity = 0.3;
    let columns = 0;
    let rows = 0;
    let opacities = new Float32Array(0);
    let color = getComputedStyle(canvas).color;
    let inView = false;
    let frame = 0;
    let lastTime = 0;

    function draw() {
      if (!canvas || !context) return;
      context.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
      context.fillStyle = color;
      for (let column = 0; column < columns; column++) {
        for (let row = 0; row < rows; row++) {
          context.globalAlpha = opacities[column * rows + row];
          context.fillRect(column * spacing, row * spacing, squareSize, squareSize);
        }
      }
      context.globalAlpha = 1;
    }

    function resize() {
      if (!canvas || !context) return;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(canvas.clientWidth * dpr);
      canvas.height = Math.round(canvas.clientHeight * dpr);
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
      columns = Math.ceil(canvas.clientWidth / spacing);
      rows = Math.ceil(canvas.clientHeight / spacing);
      opacities = Float32Array.from(
        { length: columns * rows },
        () => Math.random() * maxOpacity,
      );
      draw();
    }

    function animate(time: number) {
      const elapsed = time - lastTime;
      // A subtle background does not need a full 60 fps redraw.
      if (elapsed >= 1000 / 30) {
        const chance = 0.3 * Math.min(elapsed / 1000, 0.1);
        for (let index = 0; index < opacities.length; index++) {
          if (Math.random() < chance) {
            opacities[index] = Math.random() * maxOpacity;
          }
        }
        draw();
        lastTime = time;
      }
      frame = requestAnimationFrame(animate);
    }

    function updateAnimation() {
      cancelAnimationFrame(frame);
      if (inView && !document.hidden && !reducedMotion.matches) {
        lastTime = performance.now();
        frame = requestAnimationFrame(animate);
      }
    }

    const resizeObserver = new ResizeObserver(resize);
    const intersectionObserver = new IntersectionObserver(([entry]) => {
      inView = entry.isIntersecting;
      updateAnimation();
    });
    const themeObserver = new MutationObserver(() => {
      color = getComputedStyle(canvas).color;
      draw();
    });

    resize();
    resizeObserver.observe(canvas);
    intersectionObserver.observe(canvas);
    themeObserver.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class"],
    });
    reducedMotion.addEventListener("change", updateAnimation);
    document.addEventListener("visibilitychange", updateAnimation);

    return () => {
      cancelAnimationFrame(frame);
      resizeObserver.disconnect();
      intersectionObserver.disconnect();
      themeObserver.disconnect();
      reducedMotion.removeEventListener("change", updateAnimation);
      document.removeEventListener("visibilitychange", updateAnimation);
    };
  }, []);

  return <canvas ref={canvasRef} className="top-grid" aria-hidden="true" />;
}

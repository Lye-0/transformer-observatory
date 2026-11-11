import {
  createCore,
  heading,
  syncHeading,
  q,
  escape,
  type FoundationConfig,
  type FoundationOptions,
  type FoundationController,
} from '../core';
export const LOADER_VARIANTS = ['helix-thread'] as const;
export type LoaderVariant = (typeof LOADER_VARIANTS)[number];
export function loaderArtwork(): string {
  return `<div class="ld-helix">${Array.from({ length: 14 }, (_, i) => `<i class="ld-pair" style="--i:${i}"></i>`).join('')}</div>`;
}
export function renderLoader(o: FoundationOptions & { variant?: string }): string {
  const variant = LOADER_VARIANTS.includes(o.variant as LoaderVariant)
    ? (o.variant as LoaderVariant)
    : 'helix-thread';
  return (
    heading(o) +
    `<div class="ct-loader-stage" aria-hidden="true"><div class="ct-loader-body" data-loader-art="${variant}">${loaderArtwork()}</div></div><span class="ct-loader-status" data-loader-label role="status" aria-live="polite">${escape(o.content ?? '読み込み中…')}</span>`
  );
}
/** CSS owns continuous motion; visibility observers pause it, with no RAF or timer loop. */
export function mountLoader(
  root: HTMLElement,
  config: FoundationConfig,
  options: FoundationOptions = {},
): FoundationController {
  const c = createCore(root, config, options);
  root.classList.add('sop-motion-loader');
  if (!root.querySelector('[data-loader-art]')) root.innerHTML = renderLoader({ ...config, ...options });
  const media = matchMedia('(prefers-reduced-motion: reduce)');
  let visible = true;
  const setMotion = () =>
    (root.dataset.motionRunning = String(
      !c.dead && visible && !document.hidden && !c.options.paused && !media.matches,
    ));
  const observer =
    typeof IntersectionObserver === 'undefined'
      ? null
      : new IntersectionObserver((entries) => {
          visible = entries[0]?.isIntersecting ?? true;
          setMotion();
        });
  observer?.observe(root);
  c.on(document, 'visibilitychange', setMotion);
  c.on(media, 'change', setMotion);
  c.sync = () => {
    syncHeading(c);
    const text = c.options.content ?? '読み込み中…';
    const label = q(root, '[data-loader-label]');
    if (label.textContent !== text) label.textContent = text;
    setMotion();
  };
  c.cleanup(() => {
    observer?.disconnect();
    root.dataset.motionRunning = 'false';
  });
  c.sync('initial');
  return c;
}

import { useId, useRef, type InputHTMLAttributes } from 'react';
import { Minus, Plus } from 'lucide-react';

type Props = Omit<InputHTMLAttributes<HTMLInputElement>, 'type' | 'onChange' | 'children'> & {
  value: string | number;
  onValueChange: (value: string) => void;
};

/** Keep native number validation and stepping, with full-height, themed controls. */
export default function NumberInput({
  value,
  onValueChange,
  className = '',
  min,
  max,
  step = 1,
  disabled,
  readOnly,
  ...props
}: Props) {
  const input = useRef<HTMLInputElement>(null);
  const name = props['aria-label'] || '値';
  const locked = disabled || readOnly;
  function change(direction: number) {
    const element = input.current;
    if (!element || locked) return;
    if (direction > 0) element.stepUp();
    else element.stepDown();
    onValueChange(element.value);
  }
  return (
    <span className={`number-input ${className}`} data-disabled={disabled || undefined}>
      <button
        type="button"
        aria-label={`${name}を減らす`}
        title={`${step}減らす`}
        disabled={locked || (value !== '' && min !== undefined && Number(value) <= Number(min))}
        onPointerDown={(e) => {
          if (e.button === 0) {
            e.preventDefault();
            input.current?.focus({ preventScroll: true });
          }
        }}
        onClick={() => change(-1)}
      >
        <Minus size={15} aria-hidden="true" />
      </button>
      <input
        {...props}
        ref={input}
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        readOnly={readOnly}
        onChange={(e) => onValueChange(e.target.value)}
      />
      <button
        type="button"
        aria-label={`${name}を増やす`}
        title={`${step}増やす`}
        disabled={locked || (value !== '' && max !== undefined && Number(value) >= Number(max))}
        onPointerDown={(e) => {
          if (e.button === 0) {
            e.preventDefault();
            input.current?.focus({ preventScroll: true });
          }
        }}
        onClick={() => change(1)}
      >
        <Plus size={15} aria-hidden="true" />
      </button>
    </span>
  );
}

export function NumberField({ label, className = 'field', ...props }: Props & { label: string }) {
  const generatedId = useId();
  const id = props.id || generatedId;
  return (
    <div className={className}>
      <label htmlFor={id}>{label}</label>
      <NumberInput {...props} id={id} aria-label={props['aria-label'] || label} />
    </div>
  );
}

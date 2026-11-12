type Source = { id: string; level: number; stages: string[]; model: { fingerprint: string } };
type Candidate = {
  id: string;
  level: number;
  status: string;
  available?: boolean;
  model_fingerprint: string;
  stages?: string[];
};

export function canCompare(source: Source, candidate: Candidate): boolean {
  return (
    source.id !== candidate.id &&
    source.level >= 2 &&
    candidate.level >= 2 &&
    candidate.status === 'completed' &&
    candidate.available !== false &&
    candidate.model_fingerprint === source.model.fingerprint &&
    source.stages.length > 0 &&
    source.stages.length === candidate.stages?.length &&
    source.stages.every((stage, index) => stage === candidate.stages?.[index])
  );
}

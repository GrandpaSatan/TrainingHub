import { ArrowRight, CheckCircle, Database, Layers, MessageSquare } from "lucide-react";
import { DatasetRecord, InferenceTarget, ModelRecord } from "../api/client";

type OnboardingRibbonProps = {
  models: ModelRecord[];
  datasets: DatasetRecord[];
  activeInferenceTarget: InferenceTarget | null;
};

export function OnboardingRibbon({ models, datasets, activeInferenceTarget }: OnboardingRibbonProps) {
  const steps = [
    {
      id: "models",
      label: "Download a base model from Models",
      detail: models.length ? `${models.length} registered` : "required first",
      complete: models.length > 0,
      href: "#/models",
      icon: <Layers size={16} />,
    },
    {
      id: "datasets",
      label: "Acquire a dataset",
      detail: datasets.length ? `${datasets.length} local versions` : "needed for training",
      complete: datasets.length > 0,
      href: "#/datasets/acquire",
      icon: <Database size={16} />,
    },
    {
      id: "runtime",
      label: "Pick a runtime in Models",
      detail: activeInferenceTarget ? activeInferenceTarget.display_name : "needed for chat and cleaning",
      complete: Boolean(activeInferenceTarget),
      href: "#/models",
      icon: <MessageSquare size={16} />,
    },
  ];
  const remaining = steps.filter((step) => !step.complete);

  if (remaining.length === 0) {
    return null;
  }

  return (
    <section className="thx-onboarding" aria-label="Setup next steps">
      <div className="thx-onboarding-copy">
        <span className="thx-tag">[ SETUP · {String(remaining.length).padStart(2, "0")} OPEN ]</span>
        <strong>{remaining[0].label}</strong>
      </div>
      <div className="thx-onboarding-steps">
        {steps.map((step) => (
          <a className={`thx-onboarding-step ${step.complete ? "is-complete" : "is-open"}`} href={step.href} key={step.id}>
            <span className="thx-onboarding-icon">{step.complete ? <CheckCircle size={16} /> : step.icon}</span>
            <span>
              <strong>{step.label}</strong>
              <small>{step.detail}</small>
            </span>
            {!step.complete && <ArrowRight size={15} />}
          </a>
        ))}
      </div>
    </section>
  );
}

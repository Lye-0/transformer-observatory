from .model_discovery import inspect_model
from .model_metadata import TINY_ID, capabilities, tiny_registration


class ModelCatalog:
    def models(self):
        models = [self.current_capabilities(m) for m in [tiny_registration(), *self.store.list("models")]]
        for model in models:
            inspection = inspect_model(model["path"]) if model.get("path") else None
            model["available"] = inspection is None or inspection["status"] == "available"
            model["availability_reason"] = inspection["reason"] if inspection else "内蔵の数値検証モデル"
        return models

    def current_capabilities(self, model):
        current = capabilities(model["architecture"])
        for key in ["interventions", "tuned_lens"]:
            model["capabilities"][key] = current[key]
        if any(lens["model_fingerprint"] == model["fingerprint"] for lens in self.store.list("lenses")):
            model["capabilities"]["tuned_lens"] = "学習済みローカル重みを登録済み"
        return model

    def model(self, model_id):
        if model_id == TINY_ID:
            return self.current_capabilities(tiny_registration())
        registration = self.store.get("models", model_id)
        if not registration:
            raise ValueError("モデルが登録されていません")
        inspection = inspect_model(registration["path"])
        if inspection["status"] != "available":
            raise ValueError(f"Model Managerで保存場所を再指定してください: {inspection['reason']}")
        return self.current_capabilities(registration)

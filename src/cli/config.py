from omegaconf import OmegaConf
from src.cli.common import parser, config_from
from src.config import accumulation_steps, save_config
import json
from pathlib import Path


def main():
    cli = parser("Resolve and validate experiment configuration")
    cli.add_argument("--output")
    cli.add_argument("--resolve-budget", action="store_true",
                     help="Check train/val MRI headers and resolve compression parameters without loading VLM weights")
    cli.add_argument("--budget-report", help="Write the budget preflight JSON (requires --resolve-budget)")
    args = cli.parse_args()
    config = config_from(args)
    if args.budget_report and not args.resolve_budget:
        cli.error('--budget-report requires --resolve-budget')
    if args.resolve_budget:
        from src.data.manifest import load_manifests
        from src.models.projectors.budget import prepare_budget
        samples,_=load_manifests(config)
        report=prepare_budget(config,[*samples['train'],*samples['val']])
        if args.budget_report:
            path=Path(args.budget_report);path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(report,indent=2))
    accumulation_steps(config)
    if args.output:
        save_config(config, args.output)
    print(OmegaConf.to_yaml(config))


if __name__ == "__main__":
    main()

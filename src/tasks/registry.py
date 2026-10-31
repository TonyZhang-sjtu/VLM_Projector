def build_task(config):
    if config.task.name == 'neuroqa':
        from src.tasks.neuroqa import NeuroQA
        if config.task.prompt_version != NeuroQA.version:
            raise ValueError('Unknown NeuroQA prompt version')
        return NeuroQA()
    from src.tasks.single_sequence_findings import build_task as build_report_task
    return build_report_task(config)

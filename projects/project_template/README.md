# Project template

1. `./run.sh --init projects/<new_id>` copies this template and sets the project id.
2. Put the source video at `projects/<new_id>/input/source.mp4`. Any container FFmpeg can read works (MP4, MOV, MKV and so on). Update `project.source_video` if the name differs.
3. Edit `config.yaml`: languages, project type, target video format and sample settings.
4. Validate it: `./run.sh --project projects/<new_id>/config.yaml --validate-only`
5. Plan it: `./run.sh --project projects/<new_id>/config.yaml --plan --mode analyze`

You never need to change core code for a new project. Everything that can differ between projects is a config setting.

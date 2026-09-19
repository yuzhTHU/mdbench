from src.cli.run import get_parser


def test_top_level_cli_exposes_only_runnable_commands():
    parser = get_parser()
    for command in ('validate', 'synthetic', 'export', 'feedback'):
        args = parser.parse_args([command])
        assert args.command == command

    args = parser.parse_args([
        'run', '--algorithm', 'dummy', '--problem-file', 'problem.json',
        '--answer-file', 'answer.json'])
    assert args.command == 'run'
    assert args.problem_file.name == 'problem.json'
    assert args.answer_file.name == 'answer.json'

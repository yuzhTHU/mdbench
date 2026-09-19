from src.cli.run import get_parser


def test_top_level_cli_exposes_only_runnable_commands():
    parser = get_parser()
    for command in ('validate', 'synthetic', 'export', 'feedback'):
        args = parser.parse_args([command])
        assert args.command == command

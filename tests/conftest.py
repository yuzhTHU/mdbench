from pathlib import Path
import pytest
from src.validate_problem import load_task

@pytest.fixture
def demo():
    return load_task(Path(__file__).parents[1] / 'demo_problem.yaml', check_filename=False)

@pytest.fixture
def simple_raw():
    return {'task_name': 'Response - Variant 1', 'task_description': 'Relate input to response.',
            'mutation': 'Alter the component response.',
            'mechanism_model': [
                {'formula': 'h = 2*x + 3', 'role': 'constitutive/component relations', 'description': 'Component response.'},
                {'formula': 'y = h*x', 'role': 'conservation/balance relations', 'description': 'Output balance.'}],
            'phenomenal_model': 'y = (2*x+3)*x',
            'variables': [
                {'name': 'y', 'description': 'Output.', 'role': 'target', 'unit': 'm^2'},
                {'name': 'x', 'description': 'Input.', 'role': 'input', 'unit': 'm',
                 'sampling': {'min': 1., 'max': 8., 'ood_boundary': 4., 'distribution': 'uniform'}},
                {'name': 'h', 'description': 'Hidden response.', 'role': 'internal', 'unit': 'm'}],
            'mechanism_probes': [{'probe': 'h', 'description': 'Hidden response.', 'answer': 'h'}]}

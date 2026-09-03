import numpy as np
import json
import nd2py as nd

from src.synthetic_data import generate_synthetic_data
from src.features.io import load_problem


def test_generate_synthetic_data():
    problem = load_problem("demo_problem.yaml")
    data = generate_synthetic_data(
        problem,
        seed=123,
        train_samples=64,
        id_test_samples=32,
        ood_test_samples=32,
        pilot_samples=512,
    )

    assert data["train"].shape == (4, 64)
    assert data["id_test"].shape == (4, 32)
    assert data["ood_test"].shape == (4, 32)
    assert data["variables"].tolist() == ["T", "a", "M", "m"]
    assert json.loads(str(data["generation_config"].item())) == {
        "seed": 123,
        "train_samples": 64,
        "id_test_samples": 32,
        "ood_test_samples": 32,
        "pilot_samples": 512,
    }
    assert all(np.isfinite(data[name]).all() for name in ("train", "id_test", "ood_test"))

    row = {name: index for index, name in enumerate(data["variables"].tolist())}
    for variable in [*problem.input_variables, *problem.auxiliary_input_variables]:
        if variable.sampling is None:
            continue
        boundary = float(variable.sampling["ood_boundary"])
        assert (data["train"][row[variable.name]] <= boundary).all()
        assert (data["id_test"][row[variable.name]] <= boundary).all()
        assert (data["ood_test"][row[variable.name]] >= boundary).all()

    values = {
        name: data["train"][row[name]]
        for name in ("a", "M")
    }
    values.update({constant.name: constant.value for constant in problem.constants})
    assert np.allclose(
        data["train"][row[problem.target_variable.name]],
        nd.parse(problem.phenomenological_formula).eval(values),
    )


def test_generation_is_reproducible():
    problem = load_problem("demo_problem.yaml")
    outputs = [
        generate_synthetic_data(
            problem,
            seed=7,
            train_samples=16,
            id_test_samples=8,
            ood_test_samples=8,
            pilot_samples=128,
        )
        for _ in range(2)
    ]

    for name in outputs[0]:
        assert np.array_equal(outputs[0][name], outputs[1][name])

from traffic_prediction.pipelines.training import train_and_register_model


def main() -> None:
    result = train_and_register_model()

    print("Training completed:")

    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
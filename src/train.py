import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO License Plate Detector")
    parser.add_argument("--data", type=str, required=False, default=r"D:\Users\Strix\Datasets\LicensePlate\data.yaml", help="Path to dataset.yaml")
    parser.add_argument("--model", type=str, default="yolo26n.pt", help="Pretrained model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--workers", type=int, default=1, help="Dataloader workers")
    parser.add_argument("--cache", action="store_true", help="Cache images")
    return parser.parse_args()


def main(args):
    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        cache=args.cache
    )

    print("Training completed.")
    print(results)


if __name__ == "__main__":
    args = parse_args()
    main(args)
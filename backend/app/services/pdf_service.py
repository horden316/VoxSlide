from pathlib import Path
import fitz
from PIL import Image


class PdfService:
    def __init__(self, width: int = 1920, height: int = 1080):
        self.width = width
        self.height = height

    def render_pages(self, pdf_path: Path, output_dir: Path) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        rendered: list[Path] = []
        document = fitz.open(pdf_path)
        try:
            for index, page in enumerate(document, start=1):
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                image.thumbnail((self.width, self.height), Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", (self.width, self.height), "white")
                left = (self.width - image.width) // 2
                top = (self.height - image.height) // 2
                canvas.paste(image, (left, top))
                output_path = output_dir / f"page-{index:04d}.png"
                canvas.save(output_path, "PNG")
                rendered.append(output_path)
        finally:
            document.close()
        return rendered

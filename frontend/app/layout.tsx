import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "VoxSlide",
  description: "Turn narrated presentation PDFs into MP4 videos",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

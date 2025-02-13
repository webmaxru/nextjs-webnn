import './globals.css';
import { Inter } from 'next/font/google';

const inter = Inter({ subsets: ['latin'] });

export const metadata = {
  title: 'WebNN using Transformers.js',
  description: 'WebNN using Transformers.js',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <link rel="manifest" href="app.webmanifest" />
      <link rel="icon" type="image/x-icon" href="icons/favicon.png" />
      <body className={inter.className}>{children}</body>
    </html>
  );
}

import './globals.css';

export const metadata = {
  title: 'Crowd Safety Review',
  description: 'Human review for detected crowd-safety indicators.',
};

export default function RootLayout({ children }) {
  return <html lang="en"><body>{children}</body></html>;
}

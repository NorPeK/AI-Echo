"use client";

import React, { useState } from 'react';

export default function Main() {
  const [youtubeLink, setYoutubeLink] = useState("");
  const [loading, setLoading]   = useState(false);
  const [arabic, setArabic]     = useState<string | null>(null);
  const [english, setEnglish]   = useState<string | null>(null);
  const [ttsSrc, setTtsSrc]     = useState<string | null>(null);
  const [error, setError]       = useState<string | null>(null);

  const sendClip = async () => {
    setLoading(true);
    setError(null);
    setArabic(null);
    setEnglish(null);
    setTtsSrc(null);

    try {
      const res = await fetch('/api/process', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ youtube_url: youtubeLink })
      });
      const data = await res.json();
      if (!res.ok) throw new Error((data as any).error || 'Failed');

      setArabic(data.arabic_polished);
      setEnglish(data.english);
      setTtsSrc(`data:audio/mp3;base64,${data.tts_audio_base64}`);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="relative flex flex-col items-center justify-center gap-8 px-6 pt-40 pb-24 text-center">
      {/* decorative blobs */}
      <span className="pointer-events-none absolute -left-16 top-24 h-40 w-40 -rotate-12 rounded-full bg-gradient-to-br from-primary to-secondary opacity-60 blur-2xl" />
      <span className="pointer-events-none absolute right-10 top-32 h-32 w-32 rotate-45 rounded-full bg-gradient-to-br from-primary to-secondary opacity-60 blur-2xl" />
      <span className="pointer-events-none absolute right-36 bottom-20 h-28 w-28 rounded-full bg-gradient-to-br from-primary to-secondary opacity-50 blur-2xl" />

      {/* hero copy */}
      <h1 className="max-w-5xl text-5xl font-extrabold leading-tight sm:text-5xl md:text-5xl">
        Examine the Potential of{" "}
        <span className="text-primary">AI Echo Commentary&nbsp;</span>
        Dubbing
      </h1>
      <p className="max-w-2xl text-lg text-gray-300 sm:text-xl">
        Unleash AI Echo&apos;s AI potential. Bringing the Excitement of Arabic Commentary to the World
      </p>

      {/* action card */}
      <div className="mt-10 w-full max-w-lg space-y-4 rounded-xl border border-white/10 bg-surface/70 p-8 backdrop-blur-md">
        <p className="mb-4 text-lg text-gray-300 text-center">
          Upload a YouTube link for Fahad Alotaibi real-time live commentary (no music) and ≤ 60s to get a polished English translation, and English voice clone of Fahad Alotaibi!
        </p>

        {/* YouTube link textarea */}
        <textarea
          placeholder="Paste YouTube link here..."
          value={youtubeLink}
          onChange={(e) => setYoutubeLink(e.target.value)}
          className="w-full rounded-md bg-black/30 border border-white/10 text-white p-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary resize-none"
          rows={2}
        />

        {/* Send button */}
        <button
          onClick={sendClip}
          disabled={loading || !youtubeLink}
          className="w-full rounded-md bg-gradient-to-r from-primary to-secondary py-3 text-lg font-medium transition hover:brightness-110 disabled:opacity-50"
        >
          {loading ? "Processing…" : "Send"}
        </button>

        {error && <p className="text-red-400">{error}</p>}
      </div>

      {/* results */}
      {english && (
        <div className="mt-4 w-full max-w-2xl text-left space-y-4">
          <h2 className="text-2xl font-bold">English Translation</h2>
          <p className="whitespace-pre-wrap">{english}</p>
        </div>
      )}
      {ttsSrc && (
        <>
          <audio controls src={ttsSrc} className="mt-6" />
          <button
            onClick={() => {
              const a = document.createElement('a');
              a.href = ttsSrc;
              a.download = 'dubbed_commentary.mp3';
              a.click();
            }}
            className="mt-4 rounded-md bg-purple-500 px-4 py-2 text-white hover:bg-purple-600 transition"
          >
            Download Dubbed Audio
          </button>
        </>
      )}
    </section>
  );
}

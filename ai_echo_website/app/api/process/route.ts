import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const { youtube_url } = await request.json();

    if (!youtube_url) {
      return NextResponse.json({ error: 'youtube_url is required' }, { status: 400 });
    }

    // call your FastAPI backend
    const backendRes = await fetch(
      `${process.env.NEXT_PUBLIC_BACKEND_URL}/process`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ youtube_url })
      }
    );

    const data = await backendRes.json();
    if (!backendRes.ok) {
      return NextResponse.json({ error: data.error || backendRes.statusText }, { status: 502 });
    }

    return NextResponse.json(data);
  } catch (err: any) {
    return NextResponse.json({ error: err.message || 'Unknown error' }, { status: 500 });
  }
}

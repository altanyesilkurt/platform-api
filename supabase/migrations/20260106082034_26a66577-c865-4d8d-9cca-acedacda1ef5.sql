-- Fix function search path security issue
CREATE OR REPLACE FUNCTION public.update_updated_at_column()
RETURNS TRIGGER 
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$;

-- Migration: Add chat_contexts table for conversational context persistence

CREATE TABLE IF NOT EXISTS chat_contexts (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    chat_id UUID NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    context_type VARCHAR(20) NOT NULL CHECK (context_type IN ('pr', 'commit')),
    context_data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(chat_id)
);

-- Index for fast lookups
CREATE INDEX IF NOT EXISTS idx_chat_contexts_chat_id ON chat_contexts(chat_id);

-- Auto-update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_chat_contexts_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER chat_contexts_updated_at
    BEFORE UPDATE ON chat_contexts
    FOR EACH ROW
    EXECUTE FUNCTION update_chat_contexts_updated_at();

-- Enable RLS (adjust policies based on your auth setup)
ALTER TABLE chat_contexts ENABLE ROW LEVEL SECURITY;

-- Allow all operations for now (adjust for your auth)
CREATE POLICY "Allow all operations on chat_contexts" ON chat_contexts
    FOR ALL USING (true) WITH CHECK (true);
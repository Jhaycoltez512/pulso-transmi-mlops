import { createClient } from "@supabase/supabase-js";

const url = import.meta.env.VITE_SUPABASE_URL;
const publishableKey = import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY;

if (!url || !publishableKey) {
  throw new Error("Configure VITE_SUPABASE_URL and VITE_SUPABASE_PUBLISHABLE_KEY (see .env.example).");
}

// Read-only from here: this is the publishable key, meant to be public, and RLS only
// grants it SELECT on the tables the dashboard needs (see the add_public_read_policies migration).
export const supabase = createClient(url, publishableKey);
